import os

bind             = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers          = 1
threads          = 1
worker_class     = 'sync'
timeout          = 300        # 5 minutes — enough for cold start + inference
keepalive        = 5
max_requests     = 20         # Restart worker every 20 requests to free memory
max_requests_jitter = 3
preload_app      = False
graceful_timeout = 60
worker_tmp_dir   = '/tmp'