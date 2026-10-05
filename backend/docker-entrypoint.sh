#!/bin/sh
set -e

echo "Applying database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting gunicorn..."
# Agent runs are synchronous and can take minutes, hence the long timeout.
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers "${GUNICORN_WORKERS:-2}" \
    --threads "${GUNICORN_THREADS:-4}" \
    --timeout "${GUNICORN_TIMEOUT:-900}" \
    --access-logfile -
