FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-web.txt ./
RUN python -m pip install --no-cache-dir -r requirements-web.txt \
    && useradd --create-home --uid 10001 web \
    && mkdir /cache \
    && chown web:web /cache

COPY web_app.py solution.py psb_engine.py psb_bundles.py psb_citations.py ./
COPY web ./web
USER web
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
