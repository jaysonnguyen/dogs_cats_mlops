FROM python:3.8.13
COPY ./ /app
WORKDIR /app
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN pip install -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
# CMD ["python", "inference_onnx.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
