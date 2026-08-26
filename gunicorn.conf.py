# Gunicorn configuration file for Render deployment
# Prevents 512MB RAM SIGKILL crashes & Gunicorn 30-second worker timeouts

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
threads = 2
worker_class = 'gthread'
timeout = 120
keepalive = 5
max_requests = 100
max_requests_jitter = 10
preload_app = False
