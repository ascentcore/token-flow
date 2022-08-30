FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app/

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . /app

ENV PYTHONPATH=/app

RUN apt-get update


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload" ]