FROM python:3.11-slim
WORKDIR /app
COPY services/radar_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/radar_service/ /app/services/radar_service/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/radar_service/radar_node.py"]
