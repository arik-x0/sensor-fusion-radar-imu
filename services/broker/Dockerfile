FROM python:3.11-slim
WORKDIR /app
COPY services/broker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/broker/ /app/services/broker/
COPY common/ /app/common/
ENV PYTHONPATH=/app
CMD ["python", "services/broker/broker_server.py"]
