"""Gunicorn settings for Render and other Linux hosts.

One worker process holds the TensorFlow model in memory; threads share it.
Override with environment variables: PORT, WEB_CONCURRENCY, GUNICORN_THREADS.
Gunicorn does not run on Windows; use `python app.py` for local development.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"
timeout = 120          # first request can be slow while TensorFlow initialises
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
