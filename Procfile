release: python manage.py migrate && python manage.py collectstatic --noinput
web: gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 2 --threads 4 --timeout 300 --worker-class gthread --log-level info --access-logfile - --error-logfile -
