import os
from celery import Celery
from celery.schedules import crontab
from core.config import settings

# Initialize Celery app
# Using Redis as both message broker and result backend.
# The default local redis URL is redis://localhost:6379/0
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "echo_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['services.worker_tasks']
)

# Optional Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    # Prevent memory leaks from long running workers
    worker_max_tasks_per_child=50 
)

# Setup Celery Beat - Scheduled Tasks
# This triggers the auto-cleanup task periodically
celery_app.conf.beat_schedule = {
    'cleanup-sessions-every-hour': {
        'task': 'services.worker_tasks.scheduled_cleanup_task',
        'schedule': crontab(minute=0),  # Top of every hour
        'args': ()
    },
}
