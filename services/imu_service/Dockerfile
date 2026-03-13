FROM python:3.11-slim
WORKDIR /app
COPY services/imu_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/imu_service/ /app/services/imu_service/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/imu_service/imu_node.py"]
