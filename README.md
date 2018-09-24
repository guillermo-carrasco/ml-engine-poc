# ML Engine proof of concept

The goal of this repository is to provide a ready-to-use example on how to
use Google ML engine. For that I will use a dataset provided by [kaggle](https://www.kaggle.com/c/twitter-sentiment-analysis2).

For this code to work, you need to set the following environment variables:

    PROJECT_ID: Your main Google project ID
    BUCKET_NAME: GCS bucket in where to store the models
    REGION: Region for Compute Engine

## Run a training job locally
Define the following variables:

    TRAINING_PACKAGE_PATH="[YOUR-LOCAL-PATH-TO-TRAINING-PACKAGE]/trainier/"
    MAIN_TRAINER_MODULE=trainer.train