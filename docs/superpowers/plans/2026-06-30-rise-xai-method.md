# RISE / Pluggable XAI Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** XAI デモに説明手法の選択機能を追加し、RISE を実装する（Integrated Gradients はデフォルト維持）。

**Architecture:** `xai.py` に Factory & Registry を導入し、各手法を統一署名 `method(model, image_norm (H,W,3), class_idx) -> importance_map (H,W) in [0,1]` の関数として登録。既存 `integrated_gradients` は薄いアダプタで登録、`rise` を新規追加。`TensorFlowXAIVisualizer.generate_explanation` をレジストリ経由のディスパッチに変更し、描画（smooth + overlay）は全手法で共通。`app.py` にサイドバーの手法 selectbox を追加。

**Tech Stack:** Python 3.10 (.venv), TensorFlow 2.15, OpenCV, NumPy, Streamlit, pytest.

## Global Constraints

- 既存ブランチ `feature/cloud-run-deployment` で作業（チェックアウト済み、切り替えない）。
- 統一署名: `method(model: tf.keras.Model, image_norm: np.ndarray (H,W,3), class_idx: int) -> np.ndarray (H,W) float32 in [0,1]`。
- RISE 既定値（config）: `RISE_N=500`, `RISE_GRID=7`, `RISE_PROB=0.5`, `RISE_BATCH=64`, `RISE_SEED=42`。デフォルト手法 `DEFAULT_XAI_METHOD="Integrated Gradients"`。
- RISE は seed 固定で決定的（reproducibility 規約）。マスクは `RISE_BATCH` 枚ずつミニバッチ処理。`model(x, training=False)` で推論。
- 型ヒント + docstring、未使用 import 無し、1ファイル 200-400 行以内、`print()` 禁止、`except Exception` は手法フォールバック/UI 用途のみ（bare except 禁止）。
- 既存 `integrated_gradients(model, image, baseline, class_idx, m_steps, batch_size)` のシグネチャと既存テストは不変（後方互換）。
- テスト実行: `.venv/bin/python -m pytest ...`（`uv run` ではない。`pyproject.toml` 無し）。
- Conventional Commits。各コミットメッセージ末尾に必ず付与:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Bgo5ZKFLnQEWR8b6FjdtXk
  ```
  （各 Step では簡潔に `-m "type: ..."` と記すが、上記トレーラを必ず追加）

---

## File Structure

| ファイル | 変更 | 責務 |
|---|---|---|
| `config.py` | Modify | RISE 定数 + `DEFAULT_XAI_METHOD` 追加 |
| `xai.py` | Modify | 手法レジストリ + IG アダプタ + `rise` + `generate_explanation` ディスパッチ化 |
| `app.py` | Modify | サイドバー手法 selectbox、`_render_xai` に method を渡す |
| `tests/test_config.py` | Modify | RISE/method 定数テスト |
| `tests/test_xai.py` | Modify | レジストリ / RISE / dispatch テスト |
| `README.md` | Modify | 機能に「説明手法の選択」を1行追記 |

設計の根拠は `docs/superpowers/specs/2026-06-30-rise-xai-method-design.md` を参照。

---

## Task 1: config.py — RISE 定数と既定手法

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.DEFAULT_XAI_METHOD: str`, `config.RISE_N/RISE_GRID/RISE_BATCH/RISE_SEED: int`, `config.RISE_PROB: float`。

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_config.py` の末尾に追加:
```python
def test_rise_and_method_constants():
    assert config.DEFAULT_XAI_METHOD == "Integrated Gradients"
    assert config.RISE_N == 500
    assert config.RISE_GRID == 7
    assert config.RISE_PROB == 0.5
    assert config.RISE_BATCH == 64
    assert config.RISE_SEED == 42
    assert isinstance(config.DEFAULT_XAI_METHOD, str)
    assert isinstance(config.RISE_PROB, float)
    for name in ("RISE_N", "RISE_GRID", "RISE_BATCH", "RISE_SEED"):
        assert isinstance(getattr(config, name), int)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_rise_and_method_constants -v`
Expected: FAIL（`AttributeError: module 'config' has no attribute 'DEFAULT_XAI_METHOD'`）

- [ ] **Step 3: `config.py` に定数を追記**

`config.py` の末尾（`MODEL_CACHE_TTL` の後）に追加:
```python

# Explanation methods
DEFAULT_XAI_METHOD: str = "Integrated Gradients"

# RISE (Randomized Input Sampling for Explanation)
RISE_N: int = 500        # number of random masks
RISE_GRID: int = 7       # low-res mask grid (s x s)
RISE_PROB: float = 0.5   # probability a grid cell is on (p1)
RISE_BATCH: int = 64     # masks per model forward batch (bounds peak memory)
RISE_SEED: int = 42      # reproducibility
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(config): add RISE params and default XAI method"
```

---

## Task 2: xai.py — 手法レジストリ + IG アダプタ

**Files:**
- Modify: `xai.py`
- Test: `tests/test_xai.py`

**Interfaces:**
- Consumes: `config.DEFAULT_XAI_METHOD`, 既存 `integrated_gradients`。
- Produces:
  - `register_xai_method(name: str)` デコレータ
  - `XAI_METHODS: Dict[str, Callable]`
  - `get_xai_method(name: str) -> Callable`（未知名は DEFAULT にフォールバック）
  - `list_xai_methods() -> List[str]`（DEFAULT を先頭に）
  - 登録済み `"Integrated Gradients"` → `_ig_explain(model, image, class_idx) -> (H,W)`

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_xai.py` の末尾に追加:
```python
def test_registry_lists_default_first():
    from xai import list_xai_methods
    methods = list_xai_methods()
    assert methods[0] == "Integrated Gradients"
    assert "Integrated Gradients" in methods


def test_get_xai_method_falls_back_to_default():
    from xai import get_xai_method, XAI_METHODS
    assert callable(get_xai_method("Integrated Gradients"))
    assert get_xai_method("Nonexistent") is XAI_METHODS["Integrated Gradients"]


def test_ig_adapter_returns_normalized_map(tiny_model, sample_image):
    from xai import get_xai_method
    fn = get_xai_method("Integrated Gradients")
    hm = fn(tiny_model, sample_image, 0)
    assert hm.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_xai.py::test_registry_lists_default_first -v`
Expected: FAIL（`ImportError: cannot import name 'list_xai_methods'`）

- [ ] **Step 3: `xai.py` にレジストリと IG アダプタを実装**

(a) import 行を更新。`from typing import Tuple` を:
```python
from typing import Callable, Dict, List, Tuple
```
に変更。

(b) `import config` の直後（`integrated_gradients` 定義の前）にレジストリ基盤を追加:
```python

XAI_METHODS: Dict[str, Callable[..., np.ndarray]] = {}


def register_xai_method(name: str) -> Callable:
    """Register an XAI method under `name` in the global registry."""
    def deco(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        XAI_METHODS[name] = fn
        return fn
    return deco


def get_xai_method(name: str) -> Callable[..., np.ndarray]:
    """Return the registered method, or the default method for unknown names."""
    return XAI_METHODS.get(name, XAI_METHODS[config.DEFAULT_XAI_METHOD])


def list_xai_methods() -> List[str]:
    """Return registered method names with the default method first."""
    names = list(XAI_METHODS.keys())
    names.sort(key=lambda n: (n != config.DEFAULT_XAI_METHOD, n))
    return names
```

(c) 既存 `integrated_gradients` 関数定義の**直後**（`class TensorFlowXAIVisualizer` の前）にアダプタを追加:
```python
@register_xai_method("Integrated Gradients")
def _ig_explain(
    model: tf.keras.Model, image: np.ndarray, class_idx: int
) -> np.ndarray:
    """Registry adapter: Integrated Gradients with a zero baseline."""
    baseline = np.zeros_like(image)
    return integrated_gradients(model, image, baseline, class_idx)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/test_xai.py -v`
Expected: PASS（既存 3 + 新規 3 = 6 tests）

- [ ] **Step 5: Commit**

```bash
git add xai.py tests/test_xai.py
git commit -m "feat(xai): add pluggable XAI method registry and IG adapter"
```

---

## Task 3: xai.py — RISE 実装

**Files:**
- Modify: `xai.py`
- Test: `tests/test_xai.py`

**Interfaces:**
- Consumes: `config.RISE_*`, `register_xai_method`（Task 2）。
- Produces: 登録済み `"RISE"` → `rise(model, image, class_idx, n_masks=config.RISE_N, grid=config.RISE_GRID, prob=config.RISE_PROB, batch_size=config.RISE_BATCH, seed=config.RISE_SEED) -> np.ndarray (H,W)`。

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_xai.py` の末尾に追加:
```python
def test_rise_shape_and_range(tiny_model, sample_image):
    from xai import rise
    hm = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32)
    assert hm.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert hm.dtype == np.float32
    assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6


def test_rise_deterministic_with_seed(tiny_model, sample_image):
    from xai import rise
    a = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32, seed=123)
    b = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32, seed=123)
    assert np.allclose(a, b)


def test_rise_registered():
    from xai import list_xai_methods
    assert "RISE" in list_xai_methods()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_xai.py::test_rise_shape_and_range -v`
Expected: FAIL（`ImportError: cannot import name 'rise'`）

- [ ] **Step 3: `xai.py` に `rise` を実装**

`_ig_explain`（Task 2）の直後、`class TensorFlowXAIVisualizer` の前に追加:
```python
@register_xai_method("RISE")
def rise(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int,
    n_masks: int = config.RISE_N,
    grid: int = config.RISE_GRID,
    prob: float = config.RISE_PROB,
    batch_size: int = config.RISE_BATCH,
    seed: int = config.RISE_SEED,
) -> np.ndarray:
    """Compute a RISE saliency map via randomized input masking.

    Generates n_masks low-res (grid x grid) binary masks, bilinearly upsamples
    and randomly shifts each to (H, W), occludes the input, and weights each
    mask by the model's score for class_idx. Masks are processed in batches of
    batch_size to bound peak memory. Seeded for reproducibility.

    Args:
        model: Keras model mapping (N, H, W, 3) -> (N, num_classes).
        image: Preprocessed input image, shape (H, W, 3), float32.
        class_idx: Target class index.
        n_masks: Number of random masks.
        grid: Low-resolution mask grid size (grid x grid).
        prob: Probability a grid cell is on.
        batch_size: Masks per model forward batch.
        seed: RNG seed for reproducibility.

    Returns:
        Normalized importance map of shape (H, W), float32 in [0, 1].
    """
    rng = np.random.RandomState(seed)
    h = w = config.IMG_SIZE
    cell_h = int(np.ceil(h / grid))
    cell_w = int(np.ceil(w / grid))
    up_h = (grid + 1) * cell_h
    up_w = (grid + 1) * cell_w
    image = image.astype(np.float32)

    weighted = np.zeros((h, w), dtype=np.float32)
    for start in range(0, n_masks, batch_size):
        b = min(batch_size, n_masks - start)
        grid_masks = (rng.rand(b, grid, grid) < prob).astype(np.float32)
        masks = np.empty((b, h, w), dtype=np.float32)
        masked = np.empty((b, h, w, 3), dtype=np.float32)
        for i in range(b):
            up = cv2.resize(
                grid_masks[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR
            )
            x = rng.randint(0, up_w - w + 1)
            y = rng.randint(0, up_h - h + 1)
            m = up[y:y + h, x:x + w]
            masks[i] = m
            masked[i] = image * m[..., np.newaxis]
        preds = model(masked, training=False).numpy()
        scores = preds[:, class_idx]
        weighted += np.tensordot(scores, masks, axes=([0], [0]))

    saliency = weighted / (float(n_masks) * prob)
    saliency = (saliency - saliency.min()) / (
        saliency.max() - saliency.min() + 1e-8
    )
    return saliency.astype(np.float32)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/test_xai.py -v`
Expected: PASS（6 + 新規 3 = 9 tests）

- [ ] **Step 5: Commit**

```bash
git add xai.py tests/test_xai.py
git commit -m "feat(xai): implement RISE saliency method"
```

---

## Task 4: xai.py — generate_explanation のディスパッチ化

**Files:**
- Modify: `xai.py:63-107`（`TensorFlowXAIVisualizer` の `generate_explanation` と旧 private メソッド）
- Test: `tests/test_xai.py`

**Interfaces:**
- Consumes: `get_xai_method`（Task 2）, 登録済み IG/RISE（Task 2,3）, `integrated_gradients`。
- Produces: `TensorFlowXAIVisualizer.generate_explanation(image, class_idx, method=config.DEFAULT_XAI_METHOD) -> (overlay_bgr (224,224,3) uint8, heatmap_display (224,224) uint8)`。

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_xai.py` の末尾に追加:
```python
def test_generate_explanation_rise(tiny_model):
    from xai import TensorFlowXAIVisualizer
    viz = TensorFlowXAIVisualizer(tiny_model)
    image_bgr = (np.random.RandomState(5).rand(300, 400, 3) * 255).astype(np.uint8)
    overlay, heatmap = viz.generate_explanation(image_bgr, 0, method="RISE")
    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
    assert overlay.dtype == np.uint8
    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)


def test_generate_explanation_unknown_method_falls_back(tiny_model):
    from xai import TensorFlowXAIVisualizer
    viz = TensorFlowXAIVisualizer(tiny_model)
    image_bgr = (np.random.RandomState(6).rand(240, 240, 3) * 255).astype(np.uint8)
    # unknown method name resolves to the default (IG) and still returns a map
    overlay, heatmap = viz.generate_explanation(image_bgr, 0, method="does-not-exist")
    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/test_xai.py::test_generate_explanation_rise -v`
Expected: FAIL（`TypeError: generate_explanation() got an unexpected keyword argument 'method'`）

- [ ] **Step 3: `generate_explanation` を書き換え、旧 private メソッドを削除**

`xai.py` の `class TensorFlowXAIVisualizer` 内、現在の `generate_explanation`・`_generate_integrated_gradients`・`_generate_guided_backprop`（3 メソッド、現状 `xai.py:69-107`）を、以下の **1 メソッドに置換**（`_smooth_heatmap` / `_create_visualization` / `_preprocess_image` はそのまま残す）:
```python
    def generate_explanation(
        self,
        image: np.ndarray,
        class_idx: int,
        method: str = config.DEFAULT_XAI_METHOD,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (overlay_bgr, heatmap_display) using the named XAI method.

        Dispatches to the registered method; on any failure falls back to the
        always-available default (Integrated Gradients).
        """
        image_norm = self._preprocess_image(image)[0].numpy()
        fn = get_xai_method(method)
        try:
            heatmap = fn(self.model, image_norm, class_idx)
        except Exception:  # noqa: BLE001 - fall back to the default method (IG)
            baseline = np.zeros_like(image_norm)
            heatmap = integrated_gradients(
                self.model, image_norm, baseline, class_idx
            )
        heatmap = self._smooth_heatmap(heatmap, sigma=2.0)
        return self._create_visualization(image, heatmap)
```

- [ ] **Step 4: 全 xai テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/test_xai.py -v`
Expected: PASS（既存の `test_visualizer_generate_explanation_shapes` がデフォルト IG で通ること含め、全 11 tests）

- [ ] **Step 5: Commit**

```bash
git add xai.py tests/test_xai.py
git commit -m "refactor(xai): dispatch generate_explanation via method registry"
```

---

## Task 5: app.py — 手法 selectbox と method 受け渡し

**Files:**
- Modify: `app.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `xai.list_xai_methods`, `TensorFlowXAIVisualizer.generate_explanation(image, class_idx, method)`。

- [ ] **Step 1: import に `list_xai_methods` を追加**

`app.py` の以下の行:
```python
from xai import TensorFlowXAIVisualizer
```
を:
```python
from xai import TensorFlowXAIVisualizer, list_xai_methods
```

- [ ] **Step 2: `_render_xai` を method 対応に書き換え**

`app.py` の `_render_xai` 関数全体（現状 `def _render_xai(...)` から末尾の色説明 `st.info` まで）を以下に置換:
```python
def _render_xai(
    xai_visualizer: TensorFlowXAIVisualizer,
    image_bgr: np.ndarray,
    predicted_class: int,
    method: str,
) -> None:
    """Run the (semaphore-guarded) XAI computation and render the overlay."""
    with st.spinner(f"{method} を生成中..."):
        try:
            with _XAI_SEMAPHORE:
                overlay, heatmap = xai_visualizer.generate_explanation(
                    image_bgr, predicted_class, method
                )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay_rgb,
                caption=f"{method} 可視化結果",
                use_container_width=True,
            )
            st.subheader("🌡️ 重要度マップ")
            fig = Figure(figsize=(6, 6))
            ax = fig.subplots()
            ax.imshow(heatmap, cmap="jet")
            ax.axis("off")
            ax.set_title(f"{method} Importance Map")
            st.pyplot(fig)
        except Exception as e:  # noqa: BLE001 - surface to UI
            st.error(f"XAI可視化生成エラー: {str(e)}")
            st.info("モデルの予測処理で問題が発生した可能性があります。")
        st.info(
            "**色の意味:** 赤い領域ほどモデルが分類の判断に重要視している部分、"
            "青い領域は重要度が低い部分です。"
        )
```

- [ ] **Step 3: サイドバーに手法 selectbox を追加**

`app.py` の `main()` 内、クラス一覧ブロック（`if tm_model.class_names:` ... の expander）の直後、`col1, col2 = st.columns(2)` の直前に1行追加:
```python
    method = st.sidebar.selectbox("説明手法", list_xai_methods())

    col1, col2 = st.columns(2)
```

- [ ] **Step 4: 呼び出し側に method を渡す**

`app.py` の以下の呼び出し:
```python
            _render_xai(xai_visualizer, image_bgr, predicted_class)
```
を:
```python
            _render_xai(xai_visualizer, image_bgr, predicted_class, method)
```

- [ ] **Step 5: README に機能を1行追記**

`README.md` の「## ✨ 機能」セクション内、`- **🔍 XAI可視化**: ...` の行の直後に追加:
```markdown
- **🔀 説明手法の選択**: Integrated Gradients と RISE を切り替えて比較可能
```

- [ ] **Step 6: 検証（import + 全スイート + streamlit 起動）**

Run:
```bash
.venv/bin/python -c "import app; print('import ok')"
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8503 &
sleep 12
curl -sf http://localhost:8503/_stcore/health && echo " health ok"
pkill -f "streamlit run" || true
```
Expected: `import ok`、全テスト PASS（config 2 + xai 11 + model 5 + smoke 1 = 19 tests）、`ok health ok`。

- [ ] **Step 7: Commit**

```bash
git add app.py README.md
git commit -m "feat(app): add XAI method selector (IG / RISE) in sidebar"
```

---

## Self-Review（記入済み）

**1. Spec coverage:**
- レジストリ（register/get/list）→ Task 2 ✓ / IG アダプタ → Task 2 ✓
- RISE 実装（ミニバッチ・seed・グレー遮蔽・tensordot 重み付け）→ Task 3 ✓
- generate_explanation ディスパッチ + 旧メソッド削除 + IG フォールバック → Task 4 ✓
- config 定数（RISE_*・DEFAULT）→ Task 1 ✓
- UI selectbox + method 受け渡し → Task 5 ✓ / README → Task 5 ✓
- テスト（shape/range/determinism/registry/dispatch/fallback）→ Task 1-4 ✓
- 後方互換（既存 IG テスト・`integrated_gradients` 不変）→ Task 2-4 で維持 ✓

**2. Placeholder scan:** TODO/TBD 無し。全コード step に実コードを記載。

**3. Type consistency:**
- 統一署名 `(model, image_norm, class_idx) -> (H,W)`：`_ig_explain`（Task 2）・`rise`（Task 3）とも準拠。`rise` は config デフォルト引数付きで 3 引数呼び出し可能。
- `get_xai_method`/`list_xai_methods`/`register_xai_method`/`XAI_METHODS` は Task 2 定義、Task 3/4/5 で同名利用。
- `generate_explanation(image, class_idx, method=...)`：Task 4 定義、Task 5 で `(image_bgr, predicted_class, method)` 呼び出し一致。
- `list_xai_methods()` の default 先頭ソート → Task 2 test と Task 5 selectbox 既定が整合。

---

## Execution Handoff

実行は subagent-driven（推奨）または inline を選択。
