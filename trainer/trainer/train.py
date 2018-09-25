import datetime
import os
import subprocess
import sys

import pandas as pd

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Get service account key
subprocess.check_call(['gsutil', 'cp', 'gs://guillermo-ml-test/service.json', 'service.json'])

# Get data from BigQuery
data = pd.read_gbq('SELECT * FROM twitter_data.twits WHERE Training = true',
                   project_id='izettle-data-poc',
                   private_key='service.json')
X = data['SentimentText']
y = data['Sentiment']


# Train the model (no parameter tuning)
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, n_jobs=-1))])
lr_tfidf.fit(X, y)

# Export the classifier to a file
model = 'model.joblib'
joblib.dump(lr_tfidf, model)

# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', 'guillermo-ml-test', datetime.datetime.now().strftime('twitter_%Y%m%d_%H%M%S'), model)
subprocess.check_call(['gsutil', 'cp', model, model_path], stderr=sys.stdout)