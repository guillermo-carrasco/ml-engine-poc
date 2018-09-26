import datetime
import os
import subprocess
import sys

from sklearn.externals import joblib


class Model():
    def predict(self, samples):
        """
        Rule based model that predicts READER if the word READER is present in the string, else OTHER
        :param s: Array of strings
        :return:
        """
        return ['READER' if s.lower().count('reader') else 'OTHER' for s in samples]


model_name = 'model.joblib'
model = Model()
joblib.dump(model, model_name)

# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', 'guillermo-ml-test', datetime.datetime.now().strftime('rule_%Y%m%d_%H%M%S'), model_name)
subprocess.check_call(['gsutil', 'cp', model_name, model_path], stderr=sys.stdout)