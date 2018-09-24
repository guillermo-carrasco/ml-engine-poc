import datetime
import os
import subprocess
import sys

from google.cloud import bigquery
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Get data from BigQuery
client = bigquery.Client()
query = (
    'SELECT * FROM `twitter_data.twits`'
    'WHERE Training = true'
)
query_job = client.query(query)
rows = query_job.result()
data = rows.to_dataframe()
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