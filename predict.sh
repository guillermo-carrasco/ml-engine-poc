#!/usr/bin/env bash


run_local="local"
if [ "$1" == "$run_local" ]; then
    gcloud ml-engine local predict --model-dir=`pwd` --text-instances data/test_small.csv --framework SCIKIT_LEARN
else
    gcloud ml-engine predict --model twitter_sentiment --text-instances data/test_small.csv --version v1
fi