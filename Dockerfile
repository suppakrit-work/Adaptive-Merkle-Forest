# One AMF provider: the Storage Service and its DHT node, objects on the /data volume.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir "cryptography>=42"
COPY amf/ amf/
EXPOSE 8000
ENTRYPOINT ["python", "-m", "amf.provider.server", "--host", "0.0.0.0", "--port", "8000", "--data", "/data"]