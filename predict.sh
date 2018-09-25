#!/usr/bin/env bash

gcloud ml-engine predict --model twitter_sentiment --text-instances data/test_small.csv --version v1