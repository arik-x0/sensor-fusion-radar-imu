FROM python:3.11-slim
WORKDIR /app
COPY services/buffer_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/buffer_service/ /app/services/buffer_service/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/buffer_service/buffer_node.py"]
