# 説明手法のプラグイン化 + RISE 追加 設計 (Design Spec)

- Date: 2026-06-30
- Topic: XAI デモに説明手法の選択機能を追加し、まず RISE を実装する
- Status: Approved (design), pending implementation plan
- Branch: feature/cloud-run-deployment（既存の改良デモを継続）

## 1. 背景 / 動機

現状の XAI デモは Integrated Gradients (IG) 固定。ローカル動作確認で、転移学習 MobileNet + 黒ベースライン IG が「期待した領域をハイライトしない」ケースが確認された。分類タスクでは遮蔽ベースの **RISE** の方が直感的なヒートマップになりやすい。ユーザーは複数の説明手法を切り替えられるよう拡張したい（まず RISE）。

## 2. 目標 / 非目標

### 目標
- ユーザーが UI で説明手法（IG / RISE）を選択できる。
- 新手法を追加しやすい拡張可能なアーキテクチャ（Factory & Registry）。
- RISE を実装し、既存の Cloud Run 並行/メモリ設計を壊さない（ミニバッチでメモリ抑制、既存セマフォで直列化）。

### 非目標 (YAGNI)
- IG 以外の既存挙動の変更（IG はデフォルトのまま）。
- RISE 以外の手法（Grad-CAM, SmoothGrad 等）の今回実装（レジストリで将来追加可能にするのみ）。
- RISE のハイパラを UI から動的調整（config 固定値）。

## 3. 決定事項（合意済み）

| 項目 | 決定 |
|---|---|
| RISE マスク数 N | 500（高速重視） |
| デフォルト手法 | Integrated Gradients（RISE は選択式） |
| 拡張方式 | Factory & Registry（`@register_xai_method`） |
| 再現性 | RISE は seed 固定（`RISE_SEED`） |
| メモリ対策 | マスクを `RISE_BATCH` 枚ずつミニバッチ処理 |

## 4. アーキテクチャ

`xai.py` に統一シグネチャの手法レジストリを導入：

```
method(model: tf.keras.Model, image_norm: np.ndarray (H,W,3), class_idx: int) -> np.ndarray (H,W) in [0,1]
```

- `image_norm` は TM 正規化済み（`(pixel/127.5)-1`）の単一画像。
- 各手法は正規化済み importance map (H,W) を返す。**描画は共通**（`_smooth_heatmap` → `_create_visualization`）。

```
[app.py] selectbox("説明手法") ──method──▶ TensorFlowXAIVisualizer.generate_explanation(image, class_idx, method)
                                              │ _preprocess_image -> image_norm (H,W,3)
                                              │ fn = get_xai_method(method)
                                              │ heatmap = fn(model, image_norm, class_idx)   # IG or RISE
                                              │ heatmap = _smooth_heatmap(heatmap)
                                              ▼ _create_visualization -> (overlay_bgr, heatmap_display)
```

## 5. コンポーネント詳細

### 5.1 config.py 追加
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

### 5.2 xai.py レジストリ
```python
from typing import Callable, Dict, List

XAI_METHODS: Dict[str, Callable[..., np.ndarray]] = {}

def register_xai_method(name: str):
    def deco(fn):
        XAI_METHODS[name] = fn
        return fn
    return deco

def get_xai_method(name: str) -> Callable[..., np.ndarray]:
    return XAI_METHODS.get(name, XAI_METHODS[config.DEFAULT_XAI_METHOD])

def list_xai_methods() -> List[str]:
    # Default first, then the rest in registration order.
    names = list(XAI_METHODS.keys())
    names.sort(key=lambda n: (n != config.DEFAULT_XAI_METHOD, n))
    return names
```

### 5.3 IG をレジストリ化（既存関数は維持）
既存 `integrated_gradients(model, image, baseline, class_idx, ...)` は**そのまま残す**（テスト依存）。薄いアダプタを登録：
```python
@register_xai_method("Integrated Gradients")
def _ig_explain(model, image, class_idx):
    """Adapter: IG with a zero baseline and config defaults."""
    baseline = np.zeros_like(image)
    return integrated_gradients(model, image, baseline, class_idx)
```

### 5.4 RISE 実装
```python
@register_xai_method("RISE")
def rise(model, image, class_idx,
         n_masks=config.RISE_N, grid=config.RISE_GRID, prob=config.RISE_PROB,
         batch_size=config.RISE_BATCH, seed=config.RISE_SEED) -> np.ndarray:
    """RISE saliency: occlude with N random upsampled masks, weight by class score."""
```
アルゴリズム:
1. `rng = np.random.RandomState(seed)`。
2. low-res マスク `(b, grid, grid)` を `rng.rand(...) < prob` で生成（ミニバッチ `b ≤ batch_size`）。
3. 各マスクを `cv2.resize`（bilinear）で `(grid+1)*cell` に拡大し、`rng.randint` のオフセットで (H,W) にランダムクロップ（RISE の shift）。
4. `masked = image * mask[..., None]`（正規化空間での乗算 = 0→中間グレー遮蔽）。
5. `preds = model(masked_batch, training=False).numpy()`、`scores = preds[:, class_idx]`。
6. `weighted += Σ_i scores_i * mask_i`（`np.tensordot`）。
7. 全バッチ後 `saliency = weighted / (n_masks * prob)`、min-max 正規化して返す。

メモリ: 1バッチ `batch_size × H × W × 3` float32（≈ 38MB @64）。`model(x, training=False)` は単画像 `predict` より軽量でスレッドフレンドリ。

### 5.5 generate_explanation のディスパッチ化
```python
def generate_explanation(self, image, class_idx, method=config.DEFAULT_XAI_METHOD):
    image_norm = self._preprocess_image(image)[0].numpy()  # (H,W,3) normalized
    fn = get_xai_method(method)
    try:
        heatmap = fn(self.model, image_norm, class_idx)
    except Exception:  # noqa: BLE001 - fall back to the always-available default (IG)
        baseline = np.zeros_like(image_norm)
        heatmap = integrated_gradients(self.model, image_norm, baseline, class_idx)
    heatmap = self._smooth_heatmap(heatmap, sigma=2.0)
    return self._create_visualization(image, heatmap)
```
旧 `_generate_integrated_gradients` / `_generate_guided_backprop` private メソッドは削除（ロジックはレジストリ関数へ集約）。`_preprocess_image` は引き続き利用（ここでは `[0].numpy()` で正規化済み (H,W,3) を取り出す）。

### 5.6 app.py UI
- サイドバーに `method = st.sidebar.selectbox("説明手法", xai.list_xai_methods())`（既定は先頭 = IG）。
- `_render_xai(xai_visualizer, image_bgr, predicted_class, method)` に渡す。
- スピナー文言: `f"{method} を生成中..."`。
- 既存のセマフォ・キャッシュ・Figure 描画はそのまま。

## 6. データフロー / 後方互換
- `generate_explanation` の `method` 引数はデフォルト付き → 既存呼び出し・テストは非破壊。
- 既存 IG テスト（`integrated_gradients` 直接 + visualizer shape）は維持。

## 7. エラーハンドリング
- 未知の手法名 → `get_xai_method` が DEFAULT（IG）にフォールバック。
- 手法関数が例外 → `generate_explanation` が IG にフォールバック（UI も既存 try/except で握る）。

## 8. テスト (Definition of Done)
1. `rise` が `(IMG_SIZE, IMG_SIZE)` float32、値域 [0,1] を返す（tiny_model）。
2. `rise` が seed 固定で決定的（同一入力2回が `allclose`）。
3. レジストリ: `list_xai_methods()` が IG 先頭で `["Integrated Gradients", "RISE"]`、`get_xai_method("RISE")`/未知名フォールバックが正しい。
4. `generate_explanation(image, class_idx, method="RISE")` が overlay (224,224,3) uint8 / heatmap (224,224) を返す。
5. 既存 IG テスト全通過。

## 9. パフォーマンス / 並行性
- RISE N=500 を 64 枚ずつ ≈ 8 forward バッチ。ピークメモリはバッチ単位に限定。
- 既存 `XAI_SEMAPHORE` が重い XAI を直列化。`model(x, training=False)` で軽量推論。
- 体感: 生 Mac(arm64) なら数秒、Cloud Run(2vCPU) はやや長め。スピナーで吸収。

## 10. 触るファイル
`config.py`（+RISE 定数）/ `xai.py`（+registry +rise、generate_explanation ディスパッチ化）/ `app.py`（+selectbox）/ `tests/test_xai.py`（+RISE/registry テスト）/ `README.md`（手法選択を1行追記）。

## 11. リスク / 留意点
- RISE は正規化空間でマスク乗算（0→中間グレー遮蔽）。黒遮蔽にしたい場合は別途検討（今回はグレーで十分）。
- N=500 はノイズがやや乗る可能性。品質不足なら `RISE_N` を上げるだけで調整可能（config 一箇所）。
- `model(x, training=False)` の戻りは tf.Tensor → `.numpy()` 必須。
