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
