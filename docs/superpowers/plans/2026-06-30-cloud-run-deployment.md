# Cloud Run Deployment + Concurrency Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit XAI デモを Cloud Run へ移行し、各リクエストのメモリ/CPU ピークを下げて、イベントでの同時アクセス（数人〜十数人）でもクラッシュしないようにする。

**Architecture:** モノリシックな `app.py`（551行）を `config.py` / `model.py` / `xai.py` / `app.py` に分割。Integrated Gradients を 51 枚一括から `IG_BATCH=8` ごとのミニバッチ累積に変更してメモリピークを削減。モデルロードを `@st.cache_resource`（ハッシュキー・境界つき）でキャッシュ。Docker コンテナ化し、低 concurrency + session-affinity の Cloud Run（東京）に水平オートスケールでデプロイ。

**Tech Stack:** Python 3.11, Streamlit, TensorFlow 2.15（Linux は `tensorflow-cpu`、ローカルは `tensorflow`）, OpenCV (headless), Google Cloud Run, Docker, pytest, uv。

## Global Constraints

- Python ファイルは型ヒント・docstring を付け、1 ファイル 200-400 行以内（`~/.claude/rules/coding-style.md`）。
- `print()` デバッグを残さない。ロガー or 削除。
- ハードコードのハイパーパラメータ禁止。定数は `config.py` に集約。
- 依存バージョン: `tensorflow==2.15.0`（ローカル/Darwin）, `tensorflow-cpu==2.15.0`（Linux/Docker）, `streamlit>=1.28.0`, `opencv-python-headless>=4.8.0`。
- Cloud Run パラメータ（verbatim）: region `asia-northeast1`, `--cpu 2 --memory 4Gi --concurrency 3 --session-affinity --timeout 300 --max-instances 10 --min-instances 0 --allow-unauthenticated`, service name `xai-demo`。
- テスト・乱数は seed 固定（`tf.keras.utils.set_random_seed(0)`）。
- Conventional Commits。**各コミットメッセージの末尾に必ず以下を付与**：
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Bgo5ZKFLnQEWR8b6FjdtXk
  ```
  （以降の各 Step では簡潔さのため `-m "type: ..."` のみ記載するが、上記トレーラを必ず追加すること）
- ブランチ: `feature/cloud-run-deployment`（作成済み）。

---

## File Structure

| ファイル | 責務 | 行数目安 |
|---|---|---|
| `config.py` (新規) | 全定数（画像サイズ, IG パラメータ, TF スレッド, セマフォ, キャッシュ設定） | ~20 |
| `xai.py` (新規) | `integrated_gradients()` 関数 + `TensorFlowXAIVisualizer` クラス | ~200 |
| `model.py` (新規) | `parse_class_names`, `TeachableMachineModel`, `load_model_from_zip`, `load_model_cached` | ~120 |
| `app.py` (改修) | Streamlit UI + 起動時 TF 設定 + 同時実行セマフォ | ~180 |
| `requirements.txt` (改修) | 本番/ローカル両対応（env marker） | ~8 |
| `requirements-dev.txt` (新規) | dev 依存（`-r requirements.txt` + pytest） | ~2 |
| `Dockerfile` (新規) | コンテナビルド | ~25 |
| `.dockerignore` (新規) | ビルドコンテキスト除外 | ~15 |
| `deploy.sh` (新規) | `gcloud run deploy` 固定 | ~20 |
| `conftest.py` (新規, repo root) | pytest fixtures（repo_root, model_zip_bytes, tiny_model） | ~40 |
| `tests/test_xai.py` (新規) | IG ミニバッチ等価性 + visualizer 出力 shape | ~70 |
| `tests/test_model.py` (新規) | ラベル解析 + zip ロード + predict shape | ~60 |
| `tests/test_smoke_pipeline.py` (新規) | カメラ非依存の end-to-end 1 run | ~30 |

設計の根拠は `docs/superpowers/specs/2026-06-30-cloud-run-concurrency-design.md` を参照。

---

## Task 1: 依存関係の分割と開発環境セットアップ

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: 動作する uv 仮想環境（`.venv`）と `pytest` 実行環境。

- [ ] **Step 1: `requirements.txt` を env marker 対応に書き換え**

`requirements.txt` を以下で**全置換**：

```
streamlit>=1.28.0
tensorflow-cpu==2.15.0; platform_system == "Linux"
tensorflow==2.15.0; platform_system != "Linux"
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
```

> 本番（Cloud Run / Linux Docker）では `tensorflow-cpu`、ローカル（macOS 等）では従来どおり `tensorflow` が入る。

- [ ] **Step 2: `requirements-dev.txt` を作成**

```
-r requirements.txt
pytest>=7.4.0
```

- [ ] **Step 3: `.gitignore` にデモ固有の一時ファイルを追記**

`.gitignore` の末尾に以下ブロックを追記（既存行と重複しても害はない）：

```
# --- xai_demo runtime/temp artifacts ---
.venv/
__pycache__/
.pytest_cache/
temp_extract/
temp_keras_model_*.h5
temp_labels_*.txt
```

- [ ] **Step 4: 仮想環境を作成し dev 依存をインストール**

Run:
```bash
uv venv
uv pip install -r requirements-dev.txt
```
Expected: インストール成功。macOS arm64 で `tensorflow==2.15.0` が解決できない場合のみ、`tensorflow-macos==2.15.0` にフォールバックして再実行（その場合 `requirements-dev.txt` に追記）。

- [ ] **Step 5: import スモークで環境を検証**

Run:
```bash
uv run python -c "import tensorflow, streamlit, cv2, scipy, numpy, matplotlib, PIL; print('env ok')"
```
Expected: `env ok` が出力される（警告は許容）。

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt .gitignore
git commit -m "chore: split deps for linux/local and add dev requirements"
```

---

## Task 2: config.py（定数の集約）

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `IMG_SIZE`, `M_STEPS`, `IG_BATCH`, `TF_INTRA_THREADS`, `TF_INTER_THREADS`, `XAI_SEMAPHORE`, `MODEL_CACHE_MAX_ENTRIES`, `MODEL_CACHE_TTL`（すべて int）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_config.py`:
```python
"""Tests for config constants."""
import config


def test_config_constants_present_and_typed():
    assert config.IMG_SIZE == 224
    assert config.M_STEPS == 32
    assert config.IG_BATCH == 8
    assert config.TF_INTRA_THREADS == 2
    assert config.TF_INTER_THREADS == 1
    assert config.XAI_SEMAPHORE == 2
    assert config.MODEL_CACHE_MAX_ENTRIES == 8
    assert config.MODEL_CACHE_TTL == 1800
    for name in (
        "IMG_SIZE", "M_STEPS", "IG_BATCH", "TF_INTRA_THREADS",
        "TF_INTER_THREADS", "XAI_SEMAPHORE", "MODEL_CACHE_MAX_ENTRIES",
        "MODEL_CACHE_TTL",
    ):
        assert isinstance(getattr(config, name), int)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run python -m pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'config'`）

- [ ] **Step 3: `config.py` を実装**

```python
"""Project-wide constants for the XAI demo.

Centralizes all tunable parameters (immutable module-level constants) so no
hyperparameters are hardcoded across modules.
"""

# Image preprocessing
IMG_SIZE: int = 224

# Integrated Gradients
M_STEPS: int = 32      # number of interpolation steps (was 50)
IG_BATCH: int = 8      # interpolation images processed per gradient batch

# TensorFlow CPU threading (instance has 2 vCPUs on Cloud Run)
TF_INTRA_THREADS: int = 2
TF_INTER_THREADS: int = 1

# In-instance concurrency guard for the heavy XAI computation
XAI_SEMAPHORE: int = 2

# Streamlit resource cache bounds for per-user uploaded models
MODEL_CACHE_MAX_ENTRIES: int = 8
MODEL_CACHE_TTL: int = 1800  # seconds
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(config): add centralized constants module"
```

---

## Task 3: xai.py — ミニバッチ Integrated Gradients 関数（TDD）

**Files:**
- Create: `xai.py`
- Create: `conftest.py`（repo root, fixtures）
- Test: `tests/test_xai.py`

**Interfaces:**
- Consumes: `config.IMG_SIZE`, `config.M_STEPS`, `config.IG_BATCH`。
- Produces: `integrated_gradients(model, image, baseline, class_idx, m_steps=config.M_STEPS, batch_size=config.IG_BATCH) -> np.ndarray`。`image`/`baseline` は形状 `(IMG_SIZE, IMG_SIZE, 3)` の float32。戻り値は正規化済み importance map `(IMG_SIZE, IMG_SIZE)` float32（平滑化前）。

- [ ] **Step 1: repo root の `conftest.py` を作成（fixtures）**

`conftest.py`（リポジトリ直下に置くことで repo root が sys.path に入り、`config`/`model`/`xai` を import 可能にする）:
```python
"""Shared pytest fixtures. Placed at repo root so top-level modules import."""
import io
import os
import zipfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def model_zip_bytes() -> bytes:
    """Build an in-memory Teachable-Machine-style zip from temp_model.h5."""
    h5_path = os.path.join(REPO_ROOT, "temp_model.h5")
    with open(h5_path, "rb") as f:
        h5 = f.read()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("keras_model.h5", h5)
        zf.writestr("labels.txt", "0 ClassA\n1 ClassB\n")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_model():
    """A small deterministic Keras model for fast gradient tests."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(2, 3, padding="same")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inp, out)


@pytest.fixture()
def sample_image() -> np.ndarray:
    rng = np.random.RandomState(0)
    return (rng.rand(224, 224, 3).astype(np.float32) * 2.0) - 1.0
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_xai.py`:
```python
"""Tests for the Integrated Gradients implementation."""
import numpy as np
import tensorflow as tf

import config
from xai import integrated_gradients


def _reference_importance(model, image, baseline, class_idx, m_steps):
    """All-at-once reference implementation (original behavior)."""
    image = tf.convert_to_tensor(image, tf.float32)
    baseline = tf.convert_to_tensor(baseline, tf.float32)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    a = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    delta = image - baseline
    imgs = baseline[tf.newaxis, ...] + a * delta[tf.newaxis, ...]
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        preds = model(imgs)
        outputs = preds[:, class_idx]
    grads = tape.gradient(outputs, imgs)
    avg = tf.reduce_mean(grads, axis=0)
    integrated = (image - baseline) * avg
    importance = tf.reduce_mean(tf.abs(integrated), axis=-1)
    importance = (importance - tf.reduce_min(importance)) / (
        tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8
    )
    return importance.numpy()


def test_integrated_gradients_batched_matches_unbatched(tiny_model, sample_image):
    baseline = np.zeros_like(sample_image)
    ref = _reference_importance(tiny_model, sample_image, baseline, 1, m_steps=20)
    got = integrated_gradients(
        tiny_model, sample_image, baseline, class_idx=1, m_steps=20, batch_size=4
    )
    assert got.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert np.allclose(ref, got, atol=1e-4)


def test_integrated_gradients_output_normalized(tiny_model, sample_image):
    baseline = np.zeros_like(sample_image)
    got = integrated_gradients(tiny_model, sample_image, baseline, class_idx=0)
    assert got.min() >= 0.0
    assert got.max() <= 1.0 + 1e-6
```

- [ ] **Step 3: テストが失敗することを確認**

Run: `uv run python -m pytest tests/test_xai.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'xai'`）

- [ ] **Step 4: `xai.py` の `integrated_gradients` を実装**

`xai.py`（このタスクではファイル冒頭 + 関数のみ。クラスは Task 4 で追記）:
```python
"""TensorFlow Integrated Gradients XAI visualization."""
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from typing import Tuple

import config


def integrated_gradients(
    model,
    image: np.ndarray,
    baseline: np.ndarray,
    class_idx: int,
    m_steps: int = config.M_STEPS,
    batch_size: int = config.IG_BATCH,
) -> np.ndarray:
    """Compute a normalized Integrated Gradients importance map.

    Processes the (m_steps + 1) interpolation images in mini-batches to bound
    peak memory, accumulating gradients instead of recording one giant tape.
    Numerically equivalent to the all-at-once mean of per-step gradients.

    Args:
        model: A Keras model mapping (N, H, W, 3) -> (N, num_classes).
        image: Preprocessed input image, shape (H, W, 3), float32.
        baseline: Baseline image (e.g. zeros), same shape as image.
        class_idx: Target class index.
        m_steps: Number of interpolation steps.
        batch_size: Interpolation images per gradient batch.

    Returns:
        Normalized importance map of shape (H, W), float32 in [0, 1].
    """
    image_t = tf.convert_to_tensor(image, dtype=tf.float32)
    baseline_t = tf.convert_to_tensor(baseline, dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    num = m_steps + 1
    delta = image_t - baseline_t

    grad_sum = tf.zeros_like(image_t)
    for start in range(0, num, batch_size):
        batch_alphas = alphas[start:start + batch_size]
        a = batch_alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        imgs = baseline_t[tf.newaxis, ...] + a * delta[tf.newaxis, ...]
        with tf.GradientTape() as tape:
            tape.watch(imgs)
            preds = model(imgs)
            outputs = preds[:, class_idx]
        grads = tape.gradient(outputs, imgs)
        grad_sum += tf.reduce_sum(grads, axis=0)

    avg_grads = grad_sum / tf.cast(num, tf.float32)
    integrated = delta * avg_grads
    importance = tf.reduce_mean(tf.abs(integrated), axis=-1)
    importance = (importance - tf.reduce_min(importance)) / (
        tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8
    )
    return importance.numpy()
```

- [ ] **Step 5: テストが通ることを確認**

Run: `uv run python -m pytest tests/test_xai.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: Commit**

```bash
git add xai.py conftest.py tests/test_xai.py
git commit -m "feat(xai): add mini-batched integrated gradients function"
```

---

## Task 4: xai.py — TensorFlowXAIVisualizer クラス

**Files:**
- Modify: `xai.py`（クラスを追記）
- Test: `tests/test_xai.py`（visualizer テストを追記）

**Interfaces:**
- Consumes: `integrated_gradients`（Task 3）, `config.IMG_SIZE`。
- Produces: `TensorFlowXAIVisualizer(model)` と `.generate_explanation(image_bgr, class_idx) -> (overlay_bgr: np.ndarray (IMG_SIZE, IMG_SIZE, 3) uint8, heatmap_display: np.ndarray (IMG_SIZE, IMG_SIZE) uint8)`。

> 注: 元 `app.py` の `_find_target_conv_layer` と `_analyze_model_structure` は IG 生成に未使用の dead code（print のみ）なので移植しない。

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_xai.py` の末尾に追加:
```python
def test_visualizer_generate_explanation_shapes(tiny_model):
    from xai import TensorFlowXAIVisualizer

    viz = TensorFlowXAIVisualizer(tiny_model)
    image_bgr = (np.random.RandomState(1).rand(300, 400, 3) * 255).astype(np.uint8)
    overlay, heatmap = viz.generate_explanation(image_bgr, class_idx=0)
    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
    assert overlay.dtype == np.uint8
    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run python -m pytest tests/test_xai.py::test_visualizer_generate_explanation_shapes -v`
Expected: FAIL（`ImportError: cannot import name 'TensorFlowXAIVisualizer'`）

- [ ] **Step 3: `xai.py` に `TensorFlowXAIVisualizer` を追記**

`xai.py` の `integrated_gradients` の下に追加:
```python
class TensorFlowXAIVisualizer:
    """Integrated-Gradients-based XAI visualizer for Teachable Machine models."""

    def __init__(self, model) -> None:
        self.model = model

    def generate_explanation(
        self, image: np.ndarray, class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (overlay_bgr, heatmap_display) for the given BGR image."""
        img_array = self._preprocess_image(image)
        try:
            return self._generate_integrated_gradients(img_array, image, class_idx)
        except Exception:  # noqa: BLE001 - fall back to a cheaper method
            return self._generate_guided_backprop(img_array, image, class_idx)

    def _generate_integrated_gradients(
        self, img_array: tf.Tensor, original_image: np.ndarray, class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        baseline = tf.zeros_like(img_array[0])
        heatmap = integrated_gradients(
            self.model, img_array[0], baseline, class_idx
        )
        heatmap = self._smooth_heatmap(heatmap, sigma=2.0)
        return self._create_visualization(original_image, heatmap)

    def _generate_guided_backprop(
        self, img_array: tf.Tensor, original_image: np.ndarray, class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            predictions = self.model(img_array)
            class_output = predictions[0][class_idx]
        grads = tape.gradient(class_output, img_array)
        if grads is None:
            raise ValueError("Could not compute gradients")
        grads = tf.abs(grads[0])
        importance = tf.reduce_mean(grads, axis=-1)
        importance = (importance - tf.reduce_min(importance)) / (
            tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8
        )
        heatmap = self._smooth_heatmap(importance.numpy(), sigma=2.0)
        return self._create_visualization(original_image, heatmap)

    def _smooth_heatmap(self, heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(heatmap, sigma=sigma)
        except ImportError:
            kernel = max(3, int(2 * np.ceil(2 * sigma) + 1))
            if kernel % 2 == 0:
                kernel += 1
            return cv2.GaussianBlur(heatmap, (kernel, kernel), sigma)

    def _create_visualization(
        self, image: np.ndarray, heatmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        pil_cropped = ImageOps.fit(
            pil_image, (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
        )
        cropped = np.asarray(pil_cropped)
        rgb_img = cropped.astype(np.float32) / 255.0

        heatmap_smooth = cv2.GaussianBlur(heatmap, (11, 11), 2.0)
        heatmap_enhanced = np.clip(np.power(heatmap_smooth, 0.7), 0, 1)
        colored_bgr = cv2.applyColorMap(
            np.uint8(255 * heatmap_enhanced), cv2.COLORMAP_JET
        )
        colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        rgb_uint8 = (rgb_img * 255).astype(np.uint8)
        colored_uint8 = (colored * 255).astype(np.uint8)
        alpha = 0.6
        overlay = cv2.addWeighted(rgb_uint8, 1 - alpha, colored_uint8, alpha, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        heatmap_display = np.uint8(255 * heatmap)
        return overlay_bgr, heatmap_display

    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        pil_image = ImageOps.fit(
            pil_image, (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
        )
        image_array = np.asarray(pil_image)
        normalized = (image_array.astype(np.float32) / 127.5) - 1
        img_array = np.expand_dims(normalized, axis=0)
        return tf.convert_to_tensor(img_array)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run python -m pytest tests/test_xai.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**

```bash
git add xai.py tests/test_xai.py
git commit -m "feat(xai): add TensorFlowXAIVisualizer using batched IG, drop dead conv search"
```

---

## Task 5: model.py — モデル読み込み・キャッシュ・ラベル解析

**Files:**
- Create: `model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `config.MODEL_CACHE_MAX_ENTRIES`, `config.MODEL_CACHE_TTL`, `config.IMG_SIZE`。
- Produces:
  - `parse_class_names(labels_text: str) -> list[str]`
  - `TeachableMachineModel(model, class_names: list[str] | None)` と `.predict(image_bgr) -> (np.ndarray, int)`
  - `load_model_from_zip(zip_bytes: bytes) -> TeachableMachineModel`（temp に展開→load→temp 削除）
  - `load_model_cached(file_hash: str, _zip_bytes: bytes) -> TeachableMachineModel`（`@st.cache_resource`, `file_hash` のみキャッシュキー）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_model.py`:
```python
"""Tests for model loading and label parsing."""
import os

import numpy as np

import config
from model import parse_class_names, load_model_from_zip, TeachableMachineModel


def test_parse_class_names_strips_index_prefix():
    text = "0 iPhone\n1 Galaxy Phone\n"
    assert parse_class_names(text) == ["iPhone", "Galaxy Phone"]


def test_parse_class_names_handles_single_token_and_blanks():
    text = "Cat\n\n  \nDog\n"
    assert parse_class_names(text) == ["Cat", "Dog"]


def test_load_model_from_zip_returns_model_and_cleans_temp(model_zip_bytes, repo_root):
    before = set(os.listdir(repo_root))
    tm = load_model_from_zip(model_zip_bytes)
    after = set(os.listdir(repo_root))

    assert isinstance(tm, TeachableMachineModel)
    assert tm.class_names == ["ClassA", "ClassB"]
    # input is (None, 224, 224, 3)
    assert tuple(tm.model.input_shape[1:3]) == (config.IMG_SIZE, config.IMG_SIZE)
    # no leftover temp_* files in repo root
    leaked = [f for f in (after - before) if f.startswith("temp_")]
    assert leaked == []


def test_predict_returns_probs_and_index(model_zip_bytes):
    tm = load_model_from_zip(model_zip_bytes)
    image_bgr = (np.random.RandomState(2).rand(240, 320, 3) * 255).astype(np.uint8)
    probs, idx = tm.predict(image_bgr)
    assert probs.ndim == 1
    assert probs.shape[0] >= 2
    assert 0 <= idx < probs.shape[0]
    assert np.all(np.isfinite(probs))
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run python -m pytest tests/test_model.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'model'`）

- [ ] **Step 3: `model.py` を実装**

```python
"""Teachable Machine model loading, caching, and inference."""
import hashlib
import os
import tempfile
import zipfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

import config


def parse_class_names(labels_text: str) -> List[str]:
    """Parse labels.txt content (e.g. '0 iPhone') into class names."""
    names: List[str] = []
    for line in labels_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        names.append(parts[1] if len(parts) >= 2 else parts[0])
    return names


class TeachableMachineModel:
    """Wraps an already-loaded Keras model with TM-style preprocessing."""

    def __init__(self, model, class_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.class_names = class_names

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Predict class probabilities for a BGR image array."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        pil_image = ImageOps.fit(
            pil_image, (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
        )
        image_array = np.asarray(pil_image)
        normalized = (image_array.astype(np.float32) / 127.5) - 1
        image_batch = np.expand_dims(normalized, axis=0)
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        return predictions[0], predicted_class


def load_model_from_zip(zip_bytes: bytes) -> TeachableMachineModel:
    """Extract keras_model.h5 (+ optional labels.txt) and load into memory.

    Uses a TemporaryDirectory so no temp files leak after the model is loaded.
    """
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        model_path: Optional[str] = None
        labels_text: Optional[str] = None
        for root, _dirs, files in os.walk(extract_dir):
            for name in files:
                if name == "keras_model.h5":
                    model_path = os.path.join(root, name)
                elif name == "labels.txt":
                    with open(os.path.join(root, name), "r", encoding="utf-8") as lf:
                        labels_text = lf.read()

        if not model_path:
            raise FileNotFoundError("keras_model.h5 not found in zip file")
        model = tf.keras.models.load_model(model_path, compile=False)

    class_names = parse_class_names(labels_text) if labels_text else None
    return TeachableMachineModel(model, class_names)


@st.cache_resource(
    max_entries=config.MODEL_CACHE_MAX_ENTRIES,
    ttl=config.MODEL_CACHE_TTL,
    show_spinner=False,
)
def load_model_cached(file_hash: str, _zip_bytes: bytes) -> TeachableMachineModel:
    """Cached model load. Keyed only on file_hash (_zip_bytes is not hashed)."""
    return load_model_from_zip(_zip_bytes)


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `uv run python -m pytest tests/test_model.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat(model): add cached zip model loading and label parsing"
```

---

## Task 6: app.py の改修（UI + 起動設定 + セマフォ）と統合スモークテスト

**Files:**
- Modify: `app.py`（全面書き換え）
- Test: `tests/test_smoke_pipeline.py`

**Interfaces:**
- Consumes: `config`, `model.load_model_cached`, `model.hash_bytes`, `xai.TensorFlowXAIVisualizer`。
- Produces: `configure_tensorflow()`（idempotent）, `main()`。

- [ ] **Step 1: カメラ非依存の統合スモークテストを書く（失敗する）**

`tests/test_smoke_pipeline.py`:
```python
"""End-to-end pipeline smoke test (camera-independent)."""
import numpy as np

import config
from model import load_model_from_zip
from xai import TensorFlowXAIVisualizer


def test_full_pipeline_predict_then_explain(model_zip_bytes):
    tm = load_model_from_zip(model_zip_bytes)
    image_bgr = (np.random.RandomState(3).rand(480, 640, 3) * 255).astype(np.uint8)

    probs, idx = tm.predict(image_bgr)
    assert 0 <= idx < probs.shape[0]

    viz = TensorFlowXAIVisualizer(tm.model)
    overlay, heatmap = viz.generate_explanation(image_bgr, idx)
    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `uv run python -m pytest tests/test_smoke_pipeline.py -v`
Expected: PASS することを狙うが、この時点では Task 4/5 が完了していれば **PASS**（依存は既存モジュールのみ）。もし app.py 改修前で import エラーが出る場合は次の Step で解消。
> 注: このスモークテストは app.py に依存しないため Task 4/5 完了済みなら緑になる。app.py の改修自体は手動 verify（Step 5）で担保する。

- [ ] **Step 3: `app.py` を全面書き換え**

`app.py` を以下で**全置換**：
```python
"""Streamlit XAI demo: Teachable Machine + Integrated Gradients.

UI layer only. Model loading lives in model.py, XAI in xai.py, constants in
config.py. Designed for Cloud Run with low per-instance concurrency.
"""
import threading

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

import config
from model import load_model_cached, hash_bytes
from xai import TensorFlowXAIVisualizer

# In-instance guard so concurrent sessions don't all spike memory at once.
_XAI_SEMAPHORE = threading.BoundedSemaphore(config.XAI_SEMAPHORE)


def configure_tensorflow() -> None:
    """Limit TF CPU threads once per process (no-op if already initialized)."""
    if st.session_state.get("_tf_configured"):
        return
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRA_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTER_THREADS)
    except RuntimeError:
        # TF runtime already initialized in this process; safe to ignore.
        pass
    st.session_state["_tf_configured"] = True


def _render_predictions(tm_model, predictions, predicted_class) -> None:
    st.subheader("🎯 予測結果")
    if tm_model.class_names and len(tm_model.class_names) > predicted_class:
        st.write(f"**予測クラス:** {tm_model.class_names[predicted_class]}")
    else:
        st.write(f"**予測クラス:** Class {predicted_class}")
    st.write(f"**信頼度:** {predictions[predicted_class]:.2%}")
    st.subheader("📊 全クラスの確率")
    for i, prob in enumerate(predictions):
        if tm_model.class_names and i < len(tm_model.class_names):
            label = tm_model.class_names[i]
        else:
            label = f"Class {i}"
        st.write(f"{label}: {prob:.2%}")


def _render_xai(xai_visualizer, image_bgr, predicted_class) -> None:
    with st.spinner("Integrated Gradientsを生成中..."):
        try:
            with _XAI_SEMAPHORE:
                overlay, heatmap = xai_visualizer.generate_explanation(
                    image_bgr, predicted_class
                )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay_rgb,
                caption="Integrated Gradients可視化結果",
                use_container_width=True,
            )
            st.subheader("🌡️ 重要度マップ")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title("Integrated Gradients Importance Map")
            st.pyplot(fig)
            plt.close()
        except Exception as e:  # noqa: BLE001 - surface to UI
            st.error(f"XAI可視化生成エラー: {str(e)}")
            st.info("モデルの予測処理で問題が発生した可能性があります。")
        st.info(
            "**色の意味:** 赤い領域ほどモデルが分類の判断に重要視している部分、"
            "青い領域は重要度が低い部分です。"
        )


def _render_instructions() -> None:
    st.info("👈 サイドバーからTeachable Machineモデル（zip）をアップロードしてください。")
    st.markdown(
        """
        ## 📝 使用方法

        1. **モデルの準備** — [Teachable Machine](https://teachablemachine.withgoogle.com/)
           で画像分類モデルを作成し、TensorFlow 形式（zip）でエクスポート
        2. **アプリの設定** — サイドバーから zip をアップロード
           （keras_model.h5 と labels.txt が自動で読み込まれます）
        3. **画像分類と XAI** — Webカメラで撮影すると、分類結果と
           Integrated Gradients 可視化が表示されます
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="XAI Demo - Teachable Machine + Integrated Gradients",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configure_tensorflow()

    st.title("🤖 XAI Demo: Teachable Machine + Integrated Gradients")
    st.markdown(
        "Teachable Machine で学習したモデルを使用して Web カメラ画像を分類し、"
        "Integrated Gradients で重要領域を可視化します。"
    )

    with st.expander("🔒 プライバシーについて"):
        st.markdown(
            """
            - アップロードされた画像はサーバーに保存されません
            - モデルファイルは一時的にのみ処理され、推論後にメモリ上でのみ保持されます
            - 撮影された画像は分析後にメモリから削除されます
            - 個人情報は収集されません
            """
        )

    st.sidebar.header("モデル設定")
    uploaded_zip = st.sidebar.file_uploader(
        "Teachable Machineモデル(.zip)をアップロード",
        type=["zip"],
        help="keras_model.h5 と labels.txt を含む zip をアップロードしてください",
    )

    if uploaded_zip is None:
        _render_instructions()
        return

    try:
        zip_bytes = uploaded_zip.getvalue()
        with st.spinner("モデルを読み込み中..."):
            tm_model = load_model_cached(hash_bytes(zip_bytes), zip_bytes)
        xai_visualizer = TensorFlowXAIVisualizer(tm_model.model)
        st.success("モデルが正常に読み込まれました！")
    except Exception as e:  # noqa: BLE001 - surface to UI
        st.error(f"モデルの読み込みでエラーが発生しました: {str(e)}")
        st.info(
            "Teachable Machine でエクスポートした zip（keras_model.h5 と "
            "labels.txt を含む）を使用してください。"
        )
        return

    if tm_model.class_names:
        st.info(f"クラス数: {len(tm_model.class_names)} クラス")
        with st.expander("クラス一覧"):
            for i, name in enumerate(tm_model.class_names):
                st.write(f"{i}: {name}")

    col1, col2 = st.columns(2)
    with col1:
        st.header("📷 Webカメラ")
        camera_input = st.camera_input("写真を撮影してください")
        image_bgr = None
        predicted_class = None
        if camera_input is not None:
            image = Image.open(camera_input)
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            predictions, predicted_class = tm_model.predict(image_bgr)
            _render_predictions(tm_model, predictions, predicted_class)

    with col2:
        st.header("🔍 XAI可視化")
        if camera_input is not None and image_bgr is not None:
            _render_xai(xai_visualizer, image_bgr, predicted_class)
        else:
            st.info("Webカメラで写真を撮影すると、XAI可視化結果が表示されます。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テスト一式が通ることを確認**

Run: `uv run python -m pytest tests/ -v`
Expected: PASS（全テスト緑）

- [ ] **Step 5: app.py の起動を手動 verify（import + Streamlit 起動）**

Run:
```bash
uv run python -c "import app; print('import ok')"
uv run streamlit run app.py --server.headless true --server.port 8501 &
sleep 12
curl -sf http://localhost:8501/_stcore/health && echo " health ok"
kill %1
```
Expected: `import ok` と `ok health ok`。（`%1` が残る場合は `pkill -f "streamlit run"`）

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_smoke_pipeline.py
git commit -m "refactor(app): slim UI, cache model load, add tf-thread limit and xai semaphore"
```

---

## Task 7: Dockerfile と .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: `requirements.txt`, `config.py`, `model.py`, `xai.py`, `app.py`。
- Produces: ローカルで起動可能なコンテナイメージ `xai-demo`。

- [ ] **Step 1: `.dockerignore` を作成**

```
.git
.gitignore
.venv
__pycache__
*.pyc
.pytest_cache
temp_model.h5
temp_extract
temp_keras_model_*.h5
temp_labels_*.txt
tests
conftest.py
docs
.devcontainer
*.md
LICENSE
keras_predict.py
```

- [ ] **Step 2: `Dockerfile` を作成**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# opencv-python-headless runtime dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py model.py xai.py app.py ./

EXPOSE 8080

# Shell form so $PORT (set by Cloud Run) is expanded at runtime.
CMD streamlit run app.py \
    --server.port=$PORT --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false \
    --server.enableCORS=false --server.enableXsrfProtection=false
```

- [ ] **Step 3: イメージをビルド**

Run: `docker build -t xai-demo .`
Expected: `Successfully tagged xai-demo:latest`（または `naming to ... xai-demo`）。
> Docker が無い環境では本タスクをスキップし、その旨を記録して Cloud Run デプロイ時にビルドを Cloud Build に委ねる。

- [ ] **Step 4: コンテナを起動してヘルスチェック**

Run:
```bash
docker run -d --rm -p 8080:8080 -e PORT=8080 --name xai-demo-test xai-demo
sleep 15
curl -sf http://localhost:8080/_stcore/health && echo " health ok"
docker logs xai-demo-test | tail -n 20
docker stop xai-demo-test
```
Expected: `ok health ok` が表示され、ログに Streamlit 起動とエラーなし。

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(docker): containerize streamlit app for cloud run"
```

---

## Task 8: deploy.sh と README 更新

**Files:**
- Create: `deploy.sh`
- Modify: `README.md`

**Interfaces:**
- Produces: 実行可能なデプロイスクリプトとドキュメント。

- [ ] **Step 1: `deploy.sh` を作成**

```bash
#!/usr/bin/env bash
# Deploy the XAI demo to Google Cloud Run (Tokyo).
#
# Prerequisites:
#   - gcloud CLI authenticated: gcloud auth login
#   - project set:              gcloud config set project <PROJECT_ID>
#   - APIs enabled: Cloud Run, Cloud Build, Artifact Registry
#
# Event-day warm-up (avoid cold starts):
#   gcloud run services update xai-demo --region asia-northeast1 --min-instances 2
# After the event (scale to zero):
#   gcloud run services update xai-demo --region asia-northeast1 --min-instances 0
set -euo pipefail

SERVICE="xai-demo"
REGION="asia-northeast1"

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --cpu 2 --memory 4Gi \
  --concurrency 3 \
  --session-affinity \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --allow-unauthenticated
```

- [ ] **Step 2: 実行権限を付与し、構文チェック**

Run:
```bash
chmod +x deploy.sh
bash -n deploy.sh && echo "syntax ok"
```
Expected: `syntax ok`

- [ ] **Step 3: README にデプロイ節を追記**

`README.md` の「## 技術仕様」セクションの直前に、以下を挿入：
```markdown
## ☁️ Cloud Run へのデプロイ

同時アクセス（イベント等）でも安定するよう、本アプリは Google Cloud Run への
デプロイに対応しています。低 concurrency のインスタンスを水平オートスケール
させることで、各ユーザーのモデル/推論負荷を複数インスタンスに分散します。

### 前提
- `gcloud` CLI 認証済み（`gcloud auth login`）、プロジェクト設定済み
- Cloud Run / Cloud Build / Artifact Registry API 有効化済み
- Docker（ローカルビルド検証用、任意）

### デプロイ
```bash
./deploy.sh
```
東京リージョン（`asia-northeast1`）に、cpu 2 / memory 4Gi / concurrency 3 /
session-affinity / max-instances 10 / 公開（認証なし）でデプロイされます。
アイドル時はゼロスケールするためコストはほぼ発生しません。

### イベント運用（コールドスタート回避）
```bash
# 開始前: 温める
gcloud run services update xai-demo --region asia-northeast1 --min-instances 2
# 終了後: ゼロスケールに戻す
gcloud run services update xai-demo --region asia-northeast1 --min-instances 0
```
```

- [ ] **Step 4: Commit**

```bash
git add deploy.sh README.md
git commit -m "feat(deploy): add cloud run deploy script and docs"
```

---

## Self-Review（記入済み）

**1. Spec coverage:**
- コンテナ化 → Task 7 ✓ / requirements `tensorflow-cpu` → Task 1 ✓
- IG ミニバッチ化 → Task 3 ✓ / モデルハッシュキャッシュ → Task 5 ✓
- TF スレッド制限 → Task 6（`configure_tensorflow`）✓ / セマフォ → Task 6 ✓
- `m_steps` 32 / `model.summary()`・dead conv search 削除 → Task 2, 4 ✓
- Cloud Run 設定・deploy.sh・イベント運用 → Task 8 ✓
- 検証（docker 起動 + スモーク + 数値一致 + 手動）→ Task 3, 6, 7 ✓
- 軽いモジュール分割 → Task 2-6 ✓

**2. Placeholder scan:** TODO/TBD なし。全コード step は実コードを記載済み。

**3. Type consistency:** `load_model_from_zip`/`load_model_cached` は `TeachableMachineModel` を返す。`TeachableMachineModel(model, class_names)` の引数順は Task 5 定義と Task 6 利用で一致。`integrated_gradients(model, image, baseline, class_idx, m_steps, batch_size)` のシグネチャは Task 3 定義・Task 4 利用・test 利用で一致。`hash_bytes` は Task 5 定義・Task 6 利用で一致。

---

## Execution Handoff

実行は subagent-driven（推奨）または inline を選択。
