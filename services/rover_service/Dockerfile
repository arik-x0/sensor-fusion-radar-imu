FROM python:3.11-slim
WORKDIR /app
COPY services/rover_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/rover_service/ /app/services/rover_service/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/rover_service/rover_node.py"]
