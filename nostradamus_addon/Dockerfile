ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install build dependencies for LightGBM
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-scipy \
    py3-pandas \
    libgomp \
    cmake \
    make \
    g++ \
    git

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY rootfs /

# Set permissions
RUN chmod a+x /run.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD [ "/run.sh" ]
