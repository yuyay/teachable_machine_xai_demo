# Load / concurrency tests

イベント前に「複数人同時でも落ちない／壊れない」ことを検証する2層テスト。
（どちらも既定の `pytest tests/` には含まれない）

## 事前準備
```bash
uv pip install -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
```
- リポジトリ直下に `temp_model.h5`（層2 が zip を組む元のモデル）が存在すること

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
