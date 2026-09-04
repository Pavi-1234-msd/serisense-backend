import os

# Server socket configuration
bind = "0.0.0.0:" + os.environ.get("PORT", "5000")

# Single worker process to keep RAM under 512MB limit on Render free tier
workers = int(os.environ.get("WEB_CONCURRENCY", 1))
threads = int(os.environ.get("PYTHON_GET_WORKER_THREADS", 2))

# Timeout settings
timeout = 120
keepalive = 5
max_requests = 50
max_requests_jitter = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"