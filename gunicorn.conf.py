# Gunicorn configuration file for Render deployment
# Prevents 512MB RAM SIGKILL crashes & Gunicorn 30-second worker timeouts

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 1
worker_class = 'sync'
timeout = 120
keepalive = 2
max_requests = 50
max_requests_jitter = 5
preload_app = False