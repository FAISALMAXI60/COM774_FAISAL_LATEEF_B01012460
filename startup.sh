#!/bin/bash

echo "Starting FastAPI with Gunicorn..."
gunicorn -k uvicorn.workers.UvicornWorker src.app:app --timeout 120 --workers 1 --bind=0.0.0.0:8000
