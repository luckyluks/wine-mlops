FROM python:3.9-slim-buster

# Install python packages
COPY docker-requirements.txt /tmp/requirements.txt
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
RUN pip install -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

ENTRYPOINT [ "/docker-entrypoint.sh" ] 
CMD mlflow models serve -m ${MODEL_URL} --host 0.0.0.0 --port ${MODEL_PORT} --no-conda
# CMD tail -f /dev/null 