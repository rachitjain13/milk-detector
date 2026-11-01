
FROM node:22-bookworm-slim

WORKDIR /app

# Install Python and OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY ai-engine/requirements.txt ./ai-engine/requirements.txt
RUN pip install --no-cache-dir -r ai-engine/requirements.txt

# Install Node dependencies
COPY server/package*.json ./server/
RUN npm install --prefix server --omit=dev

# Copy project files
COPY ai-engine ./ai-engine
COPY server ./server

# Create upload directory
RUN mkdir -p /app/server/uploads

ENV NODE_ENV=production

EXPOSE 10000

CMD ["node", "server/index.js"]
