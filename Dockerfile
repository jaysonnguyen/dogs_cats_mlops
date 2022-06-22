FROM python:3.8.13
COPY ./ /app
WORKDIR /app

# install requirements
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install "dvc[s3]"
RUN pip install -r requirements.txt

# model dir
ARG MODEL_DIR=./models

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

# aws credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# initialize dvc
RUN dvc init --no-scm -f

# configuring remote server in dvc
RUN dvc remote add -d model-store s3://dogs-and-cats/models/

RUN cat .dvc/config 

# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the applications
RUN chmod -R 0755 $MODEL_DIR
ENTRYPOINT ["/usr/local/bin/python", "-m", "/app"]
CMD ["lambda_handler.lambda_handler"]
