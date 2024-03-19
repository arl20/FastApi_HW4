FROM python:3.9-slim

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./main.py /app
COPY ./utils.py /app
COPY ./fuctions_for_recommendation.py /app

COPY ./configs/config.ini /app/configs/config.ini

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]