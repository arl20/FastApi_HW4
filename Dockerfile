FROM python:3.9-slim

COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./main.py /app/main.py
COPY ./utils/utils.py /app/utils/utils.py
COPY ./utils/functions_for_recommendation.py /app/utils/functions_for_recommendation.py

COPY ./configs/config.ini /app/configs/config.ini

WORKDIR /app

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
