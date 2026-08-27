import os

bind      = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers   = 1
threads   = 1
worker_class = 'sync'
timeout   = 180        # ⬆️ increased from 120 to 180 seconds
keepalive = 5
max_requests        = 30   # ⬇️ reduced to prevent memory buildup
max_requests_jitter = 5
preload_app = False
graceful_timeout = 30