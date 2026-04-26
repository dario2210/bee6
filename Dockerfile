FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8066

CMD ["gunicorn", "-b", "0.0.0.0:8066", "bee6_dashboard:server", "--workers", "1", "--threads", "4", "--timeout", "300"]
