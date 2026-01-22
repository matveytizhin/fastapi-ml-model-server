# -----------------------------------------------------------------------------
# Этап 1: Сборка зависимостей (Builder Stage)
# -----------------------------------------------------------------------------
# (Этот этап остается без изменений, но я привожу его для полноты)
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git --no-install-recommends

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt


# -----------------------------------------------------------------------------
# Этап 2: Финальный образ (Final Stage)
# -----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# --- ИЗМЕНЕНИЕ ---
# Добавляем путь к исполняемым файлам pip в системный PATH
# Это нужно сделать ДО того, как мы переключимся на appuser
ENV PATH="/root/.local/bin:${PATH}"

RUN addgroup --system appuser && adduser --system --group appuser

COPY --from=builder /app/wheels /wheels
COPY requirements.txt .

# Установка зависимостей. Теперь uvicorn будет установлен в /root/.local/bin
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

COPY ./server /app/server

RUN chown -R appuser:appuser /app

USER appuser

# Теперь, когда контейнер запустится, он сможет найти uvicorn
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
