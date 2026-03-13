FROM python:3.11-slim
WORKDIR /app
COPY services/monitor_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/monitor_service/ /app/services/monitor_service/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/monitor_service/monitor_node.py"]
