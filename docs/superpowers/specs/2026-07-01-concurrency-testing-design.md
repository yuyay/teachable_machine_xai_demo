# 同時実行 事前テスト 設計 (Design Spec)

- Date: 2026-07-01
- Topic: イベント前に「複数人同時実行でも落ちない／壊れない」ことを検証する2層テスト
- Status: Approved (design), pending implementation plan
- Mode: compressed (per ~/.claude/rules/subproject-workflow-compression.md) — test tooling, defaults-forward
- Branch: 新規 `test/concurrency-suite`（main から）

## 1. 背景 / 目的

本番（Cloud Run）で「数人〜十数人同時」に耐えることを**イベント前に再現性高く検証**したい。汎用 HTTP 負荷ツール（ab/hey/k6）はステートフルな Streamlit 経路（WebSocket セッション + ファイルアップロード + カメラ）を再現せず、先日の `AxiosError`（セッション分割 → upload 400）の類を検出できない。そこで2層で検証する。

## 2. 決定事項（合意済み）

| 項目 | 決定 |
|---|---|
| 構築範囲 | 層1（計算同時実行）+ 層2（ブラウザ E2E）の両方 |
| 層2 既定の対象 | デプロイ済 Cloud Run URL（`--url` で可変） |
| 既定の同時数 N | 10（`--n` / 定数で可変） |
| 言語/ツール | Python（既存 venv 流用）、Playwright（Chromium） |

## 3. 非目標 (YAGNI)
- 本番アプリ（`app.py`/`xai.py` 等）のコード変更はしない（テスト追加のみ）。
- CI 統合・継続的負荷監視は対象外（手動実行のワンショット検証）。
- 厳密なベンチマーク数値の SLA 化はしない（合否は「落ちない/エラーが出ない」）。

## 4. 層1 — 計算同時実行ハーネス

- ファイル: `tests/load/test_concurrency_compute.py`（`@pytest.mark.load` で opt-in、通常 `pytest` には非含有）。
- 流れ:
  1. conftest の zip ヘルパで `temp_model.h5` から `TeachableMachineModel` をロード（1体共有）。
  2. `N`（既定 10）スレッドを起動。各スレッドは `tm.predict(img)` と `viz.generate_explanation(img, idx, method)` を **IG と RISE 両方**で数反復実行。
  3. 実行中の**ピーク RSS**を `psutil`（バックグラウンドのサンプラスレッド）で測定。
- 合格基準: 例外ゼロ・全スレッド完了・ピーク RSS < `LOAD_RSS_LIMIT_GB`（既定 3.0GB、4Gi インスタンス内）。
- 役割: アプリ内 `XAI_SEMAPHORE`（同時 XAI=2）+ 境界つきモデルキャッシュが効き、重い計算が N 並列でも破綻しないことを保証。
- 追加 dev 依存: `psutil`。

## 5. 層2 — Playwright 多セッション E2E

- ファイル: `loadtest/browser_load.py`（スタンドアロン、pytest 非含有）+ `loadtest/README.md`。
- 実行: `.venv/bin/python loadtest/browser_load.py --n 10 --url https://xai-demo-ba4vygvkya-an.a.run.app`（既定 URL は本番、`--n`/`--url` 可変、`--timeout` 既定 120s）。
- 準備（実行時生成、バイナリ非コミット）:
  - **擬似カメラ映像**: 小さな `.y4m`（数フレームのノイズ/単色、224x224 程度）を生成し、Chromium 起動フラグ `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=<path>`（+ `--use-fake-ui-for-media-stream`）で `getUserMedia` に流す。
  - **TM エクスポート zip**: `temp_model.h5` を `keras_model.h5`、`labels.txt`（`0 ClassA\n1 ClassB`）として一時 zip 化（conftest と同方式）。
- 各セッション（`async` で `N` 並列、`asyncio.gather`）:
  1. `page.goto(url)` → アプリ ready 待ち。
  2. ファイル input に zip を `set_input_files` → 「モデルが正常に読み込まれました」テキスト待ち。
  3. カメラの「写真を撮影」ボタンを click。
  4. 予測結果（「予測クラス」）とヒートマップ画像の表示を待つ。
  5. ページ内に「AxiosError」/「エラー」文言が**無い**ことを assert。
- 出力: セッション毎の pass/fail・所要時間・捕捉エラーを集計し、`X/N passed` と失敗詳細を表示。終了コードは全 pass で 0。
- 追加 dev 依存: `playwright`（+ `playwright install chromium`）。

## 6. 共通 / ファイル構成

| ファイル | 変更 |
|---|---|
| `tests/load/__init__.py`, `tests/load/test_concurrency_compute.py` | 新規（層1） |
| `loadtest/browser_load.py`, `loadtest/README.md` | 新規（層2） |
| `requirements-dev.txt` | `psutil`, `playwright` 追加 |
| `pyproject.toml`/`pytest.ini` 不要 | `@pytest.mark.load` は `-m load` で実行、既定除外は実行時 `-m "not load"` 案内で足りる（設定ファイルは作らない） |

> 注: 既定 `pytest tests/` に層1を混ぜないため、`tests/load/` を既定収集から外す簡便策として、層1テストの先頭で `pytestmark = pytest.mark.load` を付け、実行は `pytest tests/load -m load` と明示。README に明記。

## 7. テスト中に観測する本番メトリクス（層2 と併用）
- `gcloud logging read '... httpRequest.status>=400 ...'` で 4xx/5xx（upload 400 の兆候）監視。
- Cloud Run コンソール Metrics: インスタンス数・メモリ使用率・コンテナ起動レイテンシ・リクエスト数。
- 当日想定に合わせ `--min-instances` を温めた状態で実行。

## 8. 検証 (Definition of Done)
1. 層1: `pytest tests/load -m load` が N=10 で緑（例外なし・ピーク RSS < 閾値）。
2. 層2: `browser_load.py --n 10` をデプロイ済 URL に対して実行し、`10/10 passed`（または失敗時に原因が出力される）。
3. `loadtest/README.md` に両者の実行手順とメトリクス観測手順を記載。

## 9. リスク / 留意点
- Playwright の fake-media は `.y4m` 形式に敏感（ヘッダ/解像度）。生成フォーマットを README に明記し、失敗時のデバッグ手順を残す。
- 本番 URL への E2E は実トラフィックを発生させる（軽微・課金は scale-to-zero で小）。`--min-instances` を上げてテスト後に戻す。
- N=10 を1インスタンスに収めると 2 vCPU がボトルネック → レイテンシは出るが**クラッシュしない**ことが合格条件（スループットの SLA は問わない）。
- 層2 のセレクタは Streamlit の DOM 構造に依存。テキスト（「モデルが正常に読み込まれました」「予測クラス」）ベースで頑健に。
