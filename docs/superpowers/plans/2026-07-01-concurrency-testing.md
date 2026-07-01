# Concurrency Pre-Event Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** イベント前に「複数人同時実行でも落ちない／壊れない」ことを再現性高く検証する2層テストを用意する。

**Architecture:** 層1=ローカル in-process の同時実行ハーネス（N スレッドで `predict`+IG/RISE を回しメモリ/例外を測る pytest）。層2=Playwright で N 並列ヘッドレス Chromium をデプロイ済 URL に当て、アップロード→擬似カメラ撮影→結果表示を検証するスタンドアロンスクリプト。両者は本番アプリのコードを変更せず追加する。

**Tech Stack:** Python 3.10 (.venv), pytest, psutil, Playwright (Chromium), TensorFlow 2.15。

## Global Constraints

- 既存ブランチ `test/concurrency-suite` で作業（チェックアウト済み、切り替えない）。
- 本番アプリ（`app.py`/`xai.py`/`model.py`/`config.py`）は**変更しない**（テスト/ツール追加のみ）。
- 既定の同時数 `N=10`、層2 既定 URL = `https://xai-demo-ba4vygvkya-an.a.run.app`（いずれも引数/env で可変）。
- テストは venv で実行: `.venv/bin/python -m pytest ...` / `.venv/bin/python loadtest/browser_load.py ...`。
- **配置の決定（spec からの意図的変更）**: spec は `tests/load/` + `@pytest.mark.load` を想定したが、`pytest tests/`（既定 21 tests）に重い負荷テストを混ぜないため、**両層を `loadtest/` 配下に置く**（`tests/` 配下ではないので既定収集されない。pytest 設定ファイルもマーカーも不要）。ルート `conftest.py` の fixture は `loadtest/` のテストにも適用される。
- 既定スイート不変: 実装後も `.venv/bin/python -m pytest tests/ -q` は **21 passed** のまま。
- Conventional Commits。各コミット末尾に必ず付与:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Bgo5ZKFLnQEWR8b6FjdtXk
  ```

---

## File Structure

| ファイル | 変更 | 責務 |
|---|---|---|
| `requirements-dev.txt` | Modify | `psutil`, `playwright` 追加 |
| `loadtest/test_concurrency_compute.py` | Create | 層1: 同時 predict+IG/RISE のメモリ/例外検証 pytest |
| `loadtest/browser_load.py` | Create | 層2: Playwright N 並列 E2E スクリプト |
| `loadtest/README.md` | Create | 両層の実行手順 + Cloud Run メトリクス観測手順 |

設計は `docs/superpowers/specs/2026-07-01-concurrency-testing-design.md` を参照。

---

## Task 1: 依存追加と loadtest/ 構成

**Files:**
- Modify: `requirements-dev.txt`
- Create: `loadtest/.gitkeep`

**Interfaces:**
- Produces: `psutil`, `playwright`（+ Chromium）が入った `.venv`、`loadtest/` ディレクトリ。

- [ ] **Step 1: `requirements-dev.txt` に依存を追加**

`requirements-dev.txt` を以下で**全置換**：
```
-r requirements.txt
pytest>=7.4.0
psutil>=5.9.0
playwright>=1.40.0
```

- [ ] **Step 2: loadtest ディレクトリを作成**

Run:
```bash
mkdir -p loadtest && touch loadtest/.gitkeep
```

- [ ] **Step 3: dev 依存と Chromium をインストール**

Run:
```bash
uv pip install -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
```
Expected: インストール成功（Chromium ダウンロード ≈150MB）。

- [ ] **Step 4: import スモークで検証**

Run:
```bash
.venv/bin/python -c "import psutil; import playwright; from playwright.sync_api import sync_playwright; print('deps ok')"
.venv/bin/python -m playwright --version
```
Expected: `deps ok` と Playwright のバージョンが出力される。

- [ ] **Step 5: 既定スイートが不変であることを確認**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `21 passed`（loadtest/ は未作成のテストなので影響なし）

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt loadtest/.gitkeep
git commit -m "chore(loadtest): add psutil and playwright dev deps, loadtest dir"
```

---

## Task 2: 層1 — 計算同時実行ハーネス

**Files:**
- Create: `loadtest/test_concurrency_compute.py`

**Interfaces:**
- Consumes: ルート `conftest.py` の `model_zip_bytes` fixture、`model.load_model_from_zip`、`xai.TensorFlowXAIVisualizer`、`config.IMG_SIZE`。
- Produces: pytest `test_concurrent_predict_and_explain`（`loadtest/` 配下なので既定 `pytest tests/` には含まれない）。

> 重要な前提（実装者向け）:
> - `TensorFlowXAIVisualizer.generate_explanation` は **app.py の `_XAI_SEMAPHORE` を通らない**（セマフォは UI 層にある）。本ハーネスは N 並列 XAI を**直接**回す＝本番（同時 XAI≤2）より厳しい**ワーストケース**を測る、保守的な上限テスト。
> - RSS の絶対値はこのホスト（tensorflow-macos）と Cloud Run（tensorflow-cpu/Linux）で異なるため、`LOAD_RSS_LIMIT_GB` は**ソフトガード**（既定 6.0、env で調整可）。**ハードゲートは「例外ゼロ + 全 worker 完了」**。Cloud Run の正確なメモリは層2 実行中のメトリクスで確認する。
> - RISE は既定 500 マスク。`N=10`/`iters=1` でも数分かかる（負荷テストなので想定内）。

- [ ] **Step 1: テスト（=成果物）を作成**

`loadtest/test_concurrency_compute.py`:
```python
"""Layer-1 load test: concurrent in-process predict + IG/RISE.

Run explicitly (NOT part of the default `pytest tests/` suite):
    .venv/bin/python -m pytest loadtest/test_concurrency_compute.py -v -s

Stresses the heavy XAI path under N concurrent threads to confirm it does not
crash, deadlock, or leak. Calls generate_explanation directly, which does NOT
go through app.py's _XAI_SEMAPHORE, so this is a conservative worst case (all N
XAI at once) vs production (semaphore caps concurrent XAI at 2). Absolute RSS
differs from the Cloud Run instance; the soft RSS limit is configurable and the
hard gate is: no exceptions + all workers complete.
"""
import os
import threading
import time

import numpy as np
import psutil

import config
from model import load_model_from_zip
from xai import TensorFlowXAIVisualizer

N_THREADS = int(os.environ.get("LOAD_N", "10"))
ITERATIONS = int(os.environ.get("LOAD_ITERS", "1"))
RSS_LIMIT_GB = float(os.environ.get("LOAD_RSS_LIMIT_GB", "6.0"))


def _sample_peak_rss(stop_event: threading.Event, result: dict) -> None:
    """Background sampler: record the process's peak RSS until stopped."""
    proc = psutil.Process()
    peak = 0
    while not stop_event.is_set():
        peak = max(peak, proc.memory_info().rss)
        time.sleep(0.1)
    result["peak"] = peak


def test_concurrent_predict_and_explain(model_zip_bytes):
    tm = load_model_from_zip(model_zip_bytes)
    viz = TensorFlowXAIVisualizer(tm.model)
    rng = np.random.RandomState(0)
    images = [
        (rng.rand(240, 320, 3) * 255).astype(np.uint8) for _ in range(N_THREADS)
    ]

    errors: list = []

    def worker(img: np.ndarray) -> None:
        try:
            for _ in range(ITERATIONS):
                _probs, idx = tm.predict(img)
                for method in ("Integrated Gradients", "RISE"):
                    overlay, heatmap = viz.generate_explanation(img, idx, method)
                    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
                    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
        except Exception as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(repr(exc))

    stop = threading.Event()
    result: dict = {}
    sampler = threading.Thread(
        target=_sample_peak_rss, args=(stop, result), daemon=True
    )
    sampler.start()

    threads = [threading.Thread(target=worker, args=(img,)) for img in images]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    stop.set()
    sampler.join()

    peak_gb = result.get("peak", 0) / 1e9
    print(
        f"\n[load] N={N_THREADS} iters={ITERATIONS} "
        f"elapsed={elapsed:.1f}s peak_RSS={peak_gb:.2f}GB"
    )
    assert not errors, f"{len(errors)}/{N_THREADS} workers raised: {errors[:3]}"
    assert peak_gb < RSS_LIMIT_GB, (
        f"peak RSS {peak_gb:.2f}GB exceeded soft limit {RSS_LIMIT_GB}GB "
        f"(tune via LOAD_RSS_LIMIT_GB)"
    )
```

- [ ] **Step 2: 層1 を実行して緑を確認**

Run: `.venv/bin/python -m pytest loadtest/test_concurrency_compute.py -v -s`
Expected: PASS（`[load] N=10 iters=1 ... peak_RSS=...GB` が出力。数分かかる）。
> 万一ローカルの RSS がソフト上限超過で落ちる場合のみ、`LOAD_RSS_LIMIT_GB` を上げて再実行（例 `LOAD_RSS_LIMIT_GB=10 .venv/bin/python -m pytest ...`）。例外で落ちた場合は上限調整では直さず、原因を報告すること。

- [ ] **Step 3: 既定スイートに混ざらないことを確認**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `21 passed`（`loadtest/` は収集されない）

- [ ] **Step 4: Commit**

```bash
git add loadtest/test_concurrency_compute.py
git commit -m "test(loadtest): add layer-1 concurrent compute harness"
```

---

## Task 3: 層2 — Playwright 多セッション E2E + README

**Files:**
- Create: `loadtest/browser_load.py`
- Create: `loadtest/README.md`

**Interfaces:**
- Consumes: リポジトリ直下の `temp_model.h5`、Playwright（Chromium）。
- Produces: `.venv/bin/python loadtest/browser_load.py --n <N> --url <URL>` で N 並列 E2E を実行し `X/N passed` を出力、全 pass で終了コード 0。

> 重要（実装者向け）:
> - 本スクリプトの**セレクタはデプロイ済み URL に対して実際に検証**すること（Streamlit の DOM 依存）。特に (a) ファイル input、(b) カメラの撮影ボタン（Streamlit `camera_input` の英語ラベルは通常 `Take Photo`）、(c) 成功テキスト（`モデルが正常に読み込まれました` / `予測クラス`）。違っていれば**セレクタのみ調整**してよい（構造・合否ロジックは保つ）。`webapp-testing` スキルが Playwright のツールキット。
> - 主たる合否は「タイムアウト内に結果（`予測クラス`）まで到達したか」。アップロードが AxiosError で失敗すれば成功テキストに到達せずタイムアウト＝失敗として正しく検出される。
> - カメラは Chromium の fake-media フラグ + 実行時生成の `.y4m`（単色 I420）で擬似化する。

- [ ] **Step 1: `loadtest/browser_load.py` を作成**

```python
"""Layer-2 load test: N concurrent headless browser sessions against the app.

Each session: open the app -> upload a Teachable Machine model zip -> capture a
(fake) camera photo -> wait for the prediction + heatmap -> assert no error.
Runs N sessions concurrently and reports X/N passed.

Usage:
    .venv/bin/python loadtest/browser_load.py --n 10 \
        --url https://xai-demo-ba4vygvkya-an.a.run.app
"""
import argparse
import asyncio
import os
import tempfile
import zipfile

from playwright.async_api import async_playwright

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URL = "https://xai-demo-ba4vygvkya-an.a.run.app"


def _write_fake_y4m(path: str, width: int = 224, height: int = 224, frames: int = 10) -> None:
    """Write a minimal mid-gray I420 .y4m for Chromium's fake video capture."""
    y = bytes([128]) * (width * height)
    u = bytes([128]) * ((width // 2) * (height // 2))
    v = bytes([128]) * ((width // 2) * (height // 2))
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 C420\n".encode())
        for _ in range(frames):
            f.write(b"FRAME\n")
            f.write(y + u + v)


def _build_model_zip(path: str) -> None:
    """Build a Teachable-Machine-style zip from the repo's temp_model.h5."""
    h5 = os.path.join(REPO_ROOT, "temp_model.h5")
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(h5, "keras_model.h5")
        zf.writestr("labels.txt", "0 ClassA\n1 ClassB\n")


async def _run_session(browser, url: str, zip_path: str, timeout_ms: int, idx: int) -> dict:
    res = {"idx": idx, "ok": False, "error": None}
    ctx = await browser.new_context()
    try:
        try:
            await ctx.grant_permissions(["camera"], origin=url)
        except Exception:  # noqa: BLE001 - fake-ui already auto-grants; ignore
            pass
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        # (a) upload the model zip into the (hidden) file input
        await page.locator('input[type="file"]').first.set_input_files(
            zip_path, timeout=timeout_ms
        )
        await page.get_by_text("モデルが正常に読み込まれました").wait_for(timeout=timeout_ms)
        # (b) capture a photo (verify this button name against the live app)
        await page.get_by_role("button", name="Take Photo").click(timeout=timeout_ms)
        # (c) wait for prediction result + heatmap
        await page.get_by_text("予測クラス").wait_for(timeout=timeout_ms)
        body = await page.inner_text("body")
        if "AxiosError" in body or "エラーが発生" in body:
            raise RuntimeError("error text present on page")
        res["ok"] = True
    except Exception as exc:  # noqa: BLE001 - report per-session failure
        res["error"] = repr(exc)
    finally:
        await ctx.close()
    return res


async def _main_async(n: int, url: str, timeout_ms: int, headed: bool) -> int:
    tmpdir = tempfile.mkdtemp(prefix="loadtest_")
    y4m = os.path.join(tmpdir, "fake_camera.y4m")
    zip_path = os.path.join(tmpdir, "model.zip")
    _write_fake_y4m(y4m)
    _build_model_zip(zip_path)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not headed,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={y4m}",
            ],
        )
        results = await asyncio.gather(
            *[_run_session(browser, url, zip_path, timeout_ms, i) for i in range(n)]
        )
        await browser.close()

    passed = sum(1 for r in results if r["ok"])
    print(f"\n=== {passed}/{n} sessions passed (url={url}) ===")
    for r in results:
        if not r["ok"]:
            print(f"  session {r['idx']} FAILED: {r['error']}")
    return 0 if passed == n else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent browser load test")
    parser.add_argument("--n", type=int, default=10, help="concurrent sessions")
    parser.add_argument("--url", default=DEFAULT_URL, help="target app URL")
    parser.add_argument("--timeout", type=int, default=120, help="per-step timeout (s)")
    parser.add_argument("--headed", action="store_true", help="show browsers (debug)")
    args = parser.parse_args()
    return asyncio.run(_main_async(args.n, args.url, args.timeout * 1000, args.headed))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 少人数でライブ検証（セレクタ確認）**

まず `--n 1 --headed` でセレクタが正しいか確認（必要なら (a)(b)(c) のセレクタのみ調整）：
```bash
.venv/bin/python loadtest/browser_load.py --n 1 --headed --url https://xai-demo-ba4vygvkya-an.a.run.app
```
Expected: `1/1 sessions passed`。失敗する場合は `--headed` で画面を見てセレクタ（特に撮影ボタン名）を実 DOM に合わせて修正し、再実行。

- [ ] **Step 3: 同時 N=3 で並列動作を確認**

Run:
```bash
.venv/bin/python loadtest/browser_load.py --n 3 --url https://xai-demo-ba4vygvkya-an.a.run.app
```
Expected: `3/3 sessions passed`（失敗時はセッション毎の原因が出力される）。
> 本番 URL に実トラフィックを発生させる点に留意（scale-to-zero でコストは軽微）。検証は小さい N で十分。フルの N=10 はユーザーがイベント前に実行する想定。

- [ ] **Step 4: `loadtest/README.md` を作成**

```markdown
# Load / concurrency tests

イベント前に「複数人同時でも落ちない／壊れない」ことを検証する2層テスト。
（どちらも既定の `pytest tests/` には含まれない）

## 事前準備
```bash
uv pip install -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
```

## 層1: 計算同時実行（ローカル・メモリ/例外）
N スレッドで `predict`+IG/RISE を同時に回し、例外なく完了し、ピーク RSS が
ソフト上限内であることを確認する。数分かかる。
```bash
.venv/bin/python -m pytest loadtest/test_concurrency_compute.py -v -s
# 調整: LOAD_N(既定10) / LOAD_ITERS(既定1) / LOAD_RSS_LIMIT_GB(既定6.0)
```
注: RSS の絶対値はローカル(tensorflow-macos)と Cloud Run(tensorflow-cpu)で異なる。
ハードゲートは「例外ゼロ・全完了」。本番メモリは層2 実行中のメトリクスで確認する。

## 層2: ブラウザ多セッション E2E（本番同等）
N 並列ヘッドレス Chromium で、アップロード→擬似カメラ撮影→結果表示を検証する。
```bash
.venv/bin/python loadtest/browser_load.py --n 10 --url https://xai-demo-ba4vygvkya-an.a.run.app
# --headed で画面表示（デバッグ）、--timeout で各ステップのタイムアウト(秒)
```
`X/N passed` を出力し、全 pass で終了コード 0。アップロードが AxiosError で
失敗するセッションは結果テキストに到達せずタイムアウト＝失敗として検出される。

## テスト中に観測する本番メトリクス
```bash
# 4xx/5xx（upload 400 の兆候）
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="xai-demo" AND httpRequest.status>=400' \
  --project xai-demo-501015 --freshness=10m --limit=50
```
Cloud Run コンソールの Metrics（インスタンス数・メモリ使用率・起動レイテンシ・
4xx/5xx）も併せて確認。当日想定に合わせ `--min-instances` を温めてから実行する。
```

- [ ] **Step 5: Commit**

```bash
git add loadtest/browser_load.py loadtest/README.md
git commit -m "test(loadtest): add layer-2 playwright multi-session e2e and README"
```

---

## Self-Review（記入済み）

**1. Spec coverage:**
- 層1（スレッド並列 predict+IG/RISE・peak RSS・例外）→ Task 2 ✓
- 層2（Playwright N 並列・fake camera・zip・upload→撮影→結果・X/N 出力）→ Task 3 ✓
- dev 依存（psutil/playwright）→ Task 1 ✓ / README（手順+メトリクス）→ Task 3 ✓
- 既定スイート不変（21）→ Task 1 Step5 / Task 2 Step3 ✓
- 本番アプリ非変更 → 全タスクが loadtest/ と requirements-dev のみ ✓
- 配置: spec の `tests/load/`+marker を `loadtest/` 別ディレクトリに変更（Global Constraints に意図を明記）

**2. Placeholder scan:** TODO/TBD 無し。層1 は完全コード。層2 は完全コード + 「セレクタはライブ検証で調整可」と明示（曖昧さではなく既知の DOM 依存に対する運用指示）。

**3. Type consistency:**
- `generate_explanation(img, idx, method)` の呼び出しは既存実装（`image, class_idx, method`）と一致。
- `load_model_from_zip(model_zip_bytes) -> TeachableMachineModel`、`.predict` / `TensorFlowXAIVisualizer(model)` は既存シグネチャと一致。
- `model_zip_bytes` fixture は既存 conftest 由来、`loadtest/` でも有効。

---

## Execution Handoff

実行は subagent-driven（推奨）または inline を選択。
