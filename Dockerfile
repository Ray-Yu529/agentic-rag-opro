# ---------- 前端 build ----------
FROM node:20-alpine AS web
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------- 後端 ----------
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=web /app/web/dist ./web/dist

# 前端由 FastAPI 直接服務 (server.py 掛 web/dist)，單容器一個 port 搞定
ENV HOST=0.0.0.0 \
    PORT=8000 \
    CACHE_PATH=/app/data/cache.json
EXPOSE 8000
CMD ["python", "server.py"]
