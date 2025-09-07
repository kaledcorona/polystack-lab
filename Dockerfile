FROM python:3.11-slim

# System deps (add others as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USER=app
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}
WORKDIR /workspace

# Python deps first (better cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir jupyterlab

# Copy minimal source (optional; we mostly mount during dev)
# COPY src/ /workspace/src

ENV PYTHONPATH=/workspace/src
EXPOSE 8888
USER ${USER}

# Default is Jupyter, overridden by compose's 'train' service command if needed
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--ServerApp.token="]
