FROM python:3.11-slim
WORKDIR /app
COPY services/ekf_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy this service and the common package
COPY services/ekf_service/ /app/services/ekf_service/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/ekf_service/ekf_node.py"]
