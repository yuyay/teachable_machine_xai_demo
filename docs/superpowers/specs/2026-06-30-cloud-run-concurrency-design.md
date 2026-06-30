# Cloud Run 移行 + 同時実行耐性 設計 (Design Spec)

- Date: 2026-06-30
- Topic: Streamlit XAI デモの同時実行クラッシュ対策（Cloud Run 移行 + アプリ内最適化）
- Status: Approved (design), pending implementation plan

## 1. 背景 / 問題

Streamlit Community Cloud は、アプリの **全訪問者の全セッションを単一の小さな共有コンテナ**（実質 1 プロセス・限られた RAM・約 1 vCPU）で実行する。本デモでは 2 要因が掛け算で効き、同時アクセス時にクラッシュする。

1. **各自モデルアップロード**: 各訪問者の MobileNet ベースモデルが同一プロセスのメモリに常駐 → メモリが同時ユーザー数に比例して増加。
2. **Integrated Gradients のピーク**: `TensorFlowXAIVisualizer._generate_integrated_gradients` が 51 枚の補間画像を**一括バッチ**で生成し、MobileNet 全体に対する `GradientTape` を**一度に**記録 → 1 リクエストあたりの瞬間メモリスパイクが大きい。

結果、十数人が同時アクセスするイベントでコンテナの RAM 上限を超過 → Streamlit がプロセスを kill/restart → **全員が同時にドロップ**。さらに 1 vCPU で複数の逆伝播が直列化し**タイムアウト**も発生する。

**本質的洞察**: モデルがユーザーごとなので、メモリを定数に抑えることは原理的に不可能（同時ユーザー数に比例）。よって durable な解は「1 台の最適化」ではなく「**複数インスタンスへの水平分散**」である。最適化は天井を上げるが傾きは変えられない。

## 2. 目標 / 非目標

### 目標
- 十数人（数人〜15 人程度）の同時バーストでクラッシュしない。
- イベント間（アイドル時）のコストはほぼ 0。
- 既存の XAI デモ機能（カメラ撮影 → 分類 → IG 可視化）を完全に維持。

### 非目標 (YAGNI)
- フロント/バックエンド分離（将来検討、今回はやらない）。
- 認証・ユーザー管理（公開デモ前提）。
- モデル共有キャッシュ（各自アップロード前提のため共有不可）。
- 複数 XAI 手法の追加。
- 既存 UI/UX の刷新。

## 3. 決定事項（合意済み）

| 項目 | 決定 |
|---|---|
| ホスティング | Google Cloud Run |
| リージョン | 東京 (`asia-northeast1`) |
| アクセス制御 | 公開（`--allow-unauthenticated`） |
| モデル | 各自アップロード（共有キャッシュ不可） |
| 同時利用規模 | 展示・イベント、数人〜十数人が同時（バースト） |
| コード構造 | 軽いモジュール分割（`config.py` / `model.py` / `xai.py` / `app.py`） |

## 4. 解決戦略

**低 concurrency インスタンスの水平オートスケール（分散）** + **リクエスト単位のピーク削減（最適化）** の二段構え。

- 分散: Cloud Run が同時リクエスト数に応じて 0〜N インスタンスを自動起動。`--concurrency` を小さく設定し、各インスタンスが担当する重い処理を制限 → メモリを隔離。
- 最適化: IG のミニバッチ化・モデルキャッシュ・TF スレッド制限等で 1 リクエストのピークを下げ、各インスタンスを安定させる。

## 5. アーキテクチャ

```
[Browser]
   │ HTTPS / WebSocket (session affinity)
   ▼
[Cloud Run service: xai-demo  (asia-northeast1)]
   │  autoscale 0..10 instances
   │  per instance: cpu=2 / memory=4Gi / concurrency=3
   ▼
[Container]
   - Streamlit on $PORT (=8080), 0.0.0.0, headless
   - tensorflow-cpu 2.15.0
   - in-process: model load (cached) + mini-batched Integrated Gradients
```

## 6. ファイル構成

### 新規（コンテナ/デプロイ）
- `Dockerfile` — `python:3.11-slim` ベース。Streamlit を `$PORT` / `0.0.0.0` / headless 起動。
- `.dockerignore` — `.git`, `temp_*`, `temp_model.h5`（2.4MB テスト用、本番イメージ非同梱）, `__pycache__`, `docs/` 等を除外。
- `deploy.sh` — `gcloud run deploy` を再現可能な形で固定。

### 新規（リファクタ: 軽いモジュール分割）
- `config.py` — 定数（`M_STEPS=32`, `IG_BATCH=8`, `IMG_SIZE=224`, `TF_INTRA_THREADS=2`, `TF_INTER_THREADS=1`, `XAI_SEMAPHORE=2`, `MODEL_CACHE_MAX_ENTRIES=8`, `MODEL_CACHE_TTL=1800`）。
- `model.py` — `TeachableMachineModel`, `extract_teachable_machine_files`, `load_model_cached`。
- `xai.py` — `TensorFlowXAIVisualizer`（ミニバッチ IG）。

### 変更
- `app.py` — Streamlit UI + 起動時 TF 設定。`model` / `xai` を import。各ファイル 200-400 行以内。
- `requirements.txt` — `tensorflow==2.15.0` → `tensorflow-cpu==2.15.0`。

## 7. 主要変更の詳細

### 7.1 コンテナ化（Dockerfile）
- ベース: `python:3.11-slim`。
- 依存: `pip install -r requirements.txt`。`opencv-python-headless` 前提なので GUI 系 system lib は原則不要（ビルド時に `libGL` 要否を確認）。
- 起動（shell 形式 CMD で `$PORT` 展開）:
  ```
  streamlit run app.py \
    --server.port=$PORT --server.address=0.0.0.0 \
    --server.headless=true --browser.gatherUsageStats=false \
    --server.enableCORS=false --server.enableXsrfProtection=false
  ```
- ヘルスチェック: Cloud Run は `/_stcore/health` を利用可能。

### 7.2 Integrated Gradients のミニバッチ化（xai.py）← 効果最大
51 枚一括の `GradientTape` を `IG_BATCH=8` ごとのループ + 勾配累積に変更し、ピークメモリを約 1/6 に削減。平均の線形性により元実装と数値的に等価。

```python
grad_sum = tf.zeros_like(img_array[0])
for batch_alphas in chunks(alphas, IG_BATCH):
    imgs = interpolate_images(baseline[0], img_array[0], batch_alphas)
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        preds = self.model(imgs)
        out = preds[:, class_idx]
    g = tape.gradient(out, imgs)
    grad_sum += tf.reduce_sum(g, axis=0)
avg_grads = grad_sum / tf.cast(len(alphas), tf.float32)  # = reduce_mean over all steps
integrated_grads = (img_array[0] - baseline[0]) * avg_grads
```
> 注: 元実装は `reduce_mean(grads, axis=0)`（= 全 `m_steps+1` 枚の平均）。累積版も同じ枚数で割り一致させる。

### 7.3 モデルのハッシュキャッシュ（model.py）
- `@st.cache_resource(max_entries=MODEL_CACHE_MAX_ENTRIES, ttl=MODEL_CACHE_TTL)` の `load_model_cached(file_hash, model_bytes, labels_text)` を導入。
- キー: アップロード zip の `sha256` ハッシュ。
- 処理: bytes を一時ファイルへ書き出し → `tf.keras.models.load_model` → **一時ファイル即削除** → メモリ上の model を返す。
- 効果: 写真撮影など毎回の再 run でのモデル再ロードを解消。`max_entries` / `ttl` で境界を設けメモリリークを防止。

### 7.4 起動時 TF 設定（app.py 冒頭、1 回のみ）
```python
tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_THREADS)  # 2
tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_THREADS)  # 1
```
concurrency=3 が 2 vCPU を過剰に奪い合わないようにする。`st.session_state` 等で 1 回だけ適用するガードを付ける。

### 7.5 同時実行セマフォ
- module-level `threading.BoundedSemaphore(XAI_SEMAPHORE=2)` を XAI 計算の前後で acquire/release。
- バーストのスパイク同時多発を抑える安全網。待機中はスピナー表示。Cloud Run の concurrency=3 を主、これを副の防御線とする。

### 7.6 クリーンアップ
- キャッシュ採用によりロード後 temp `.h5` を即削除。`session_state.temp_files` の管理を簡素化し、長寿命インスタンスでのファイル蓄積を防ぐ。

### 7.7 その他
- ロード毎の `model.summary()` / 全層 print（`_analyze_model_structure`）を削除またはデバッグフラグ化（CPU・ログ浪費の解消）。
- `m_steps` 50 → 32（`config.py`）。

## 8. デプロイ

### deploy.sh
```bash
gcloud run deploy xai-demo \
  --source . \
  --region asia-northeast1 \
  --cpu 2 --memory 4Gi \
  --concurrency 3 \
  --session-affinity \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --allow-unauthenticated
```

### イベント運用
- 当日開始前: `gcloud run services update xai-demo --region asia-northeast1 --min-instances 2`（温めてコールドスタート回避）。
- 終了後: `--min-instances 0` に戻す（ゼロスケールでコスト 0）。

### 事前準備
- `gcloud` CLI 認証、プロジェクト設定。
- API 有効化: Cloud Run, Cloud Build, Artifact Registry。

## 9. エラーハンドリング
- IG 失敗時は guided backprop へフォールバック（既存ロジック維持）。
- コールドスタート（ゼロスケール後の初回 10〜30s）: `--min-instances` 運用で緩和。
- デプロイ失敗: Cloud Build ログを参照。

## 10. 検証 (Definition of Done)
1. ローカル `docker build` 成功 → `docker run` で起動、`/_stcore/health` が OK、Streamlit がロードされる。
2. **カメラ非依存のスモークテスト**: サンプル画像で `TeachableMachineModel.predict` + ミニバッチ IG を端から端まで 1 回通す（実パイプライン 1 run）。
3. ミニバッチ IG が元実装と数値的に近接（`np.allclose` 許容誤差内）。
4. Cloud Run へデプロイ後、公開 URL で手動 1 回（アップロード → 撮影 → IG 表示）を確認。

## 11. コスト感
- ゼロスケールのためアイドル時 $0。
- 2 時間のイベントでピーク 10 インスタンス（2 vCPU / 4Gi）程度でも数ドル規模。`--max-instances 10` で上限固定。

## 12. リスク / 留意点
- TF イメージが大きく、ビルド/コールドスタートが遅い → `tensorflow-cpu` + `--min-instances` 運用で緩和。
- `session-affinity` でも負荷の偏りはあり得る → concurrency 低めで吸収。
- `opencv` 系 system lib → headless 版で回避するが、ビルド時に依存解決を確認。
- `@st.cache_resource` はセッション跨ぎで共有されるため、必ず `max_entries` / `ttl` で境界を設ける。

## 13. デプロイ後の訂正（2026-07-01）— concurrency は高くする

本設計は当初メモリ隔離のため `concurrency=3` を採用したが、**本番でモデルアップロード時に
`AxiosError`（`PUT /_stcore/upload_file/...` → HTTP 400）が発生**した。

根本原因: Streamlit はセッションを**インスタンスのメモリ内**に保持する。ページ読み込み時に
ブラウザは10以上のリクエスト（アセット + WebSocket）を並列で投げるため、**低 concurrency だと
1ユーザー分のバーストが複数インスタンスに分散**し、WebSocket セッションを持つインスタンスと
アップロード PUT が届くインスタンスが食い違って 400 になる（session-affinity は best-effort で
cold-burst の分散を防ぎきれない）。

訂正: **`concurrency=80` に変更**（`deploy.sh` 反映済み、本番は revision 00002 で適用・検証済み）。
メモリのピーク保護は Cloud Run の concurrency ではなく、**アプリ内の `XAI_SEMAPHORE`（同時 XAI=2）
+ 境界つきモデルキャッシュ（`max_entries`/`ttl`）**が担う、という役割分担に整理した。§7.3/§9 の
「concurrency=3 を主防御線とする」記述は本節で上書きされる。水平スケールは max-instances までの
インスタンス追加で引き続き機能する。
