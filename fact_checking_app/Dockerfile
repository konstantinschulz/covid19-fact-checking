FROM konstantinschulz/distilbert-base-uncased:latest as distilbert
FROM python:3.9-slim
WORKDIR /cfc
COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install --default-timeout=100 torch==1.9.0
RUN pip --no-cache-dir install --default-timeout=100 -r requirements.txt
COPY --from=distilbert /workspace .
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
