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
#
# NOTE on --concurrency 80 (not low): Streamlit keeps each session in instance
# memory. A single browser fires ~10+ parallel requests (assets + WebSocket) on
# load; with low concurrency those fan out across instances and the file-upload
# PUT lands on an instance without the session -> HTTP 400 (AxiosError). High
# concurrency keeps one user's requests on one instance. Per-instance memory is
# bounded by the in-app XAI semaphore + the model cache, not by Cloud Run
# concurrency.
set -euo pipefail

SERVICE="xai-demo"
REGION="asia-northeast1"

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --cpu 2 --memory 4Gi \
  --concurrency 80 \
  --session-affinity \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --allow-unauthenticated
