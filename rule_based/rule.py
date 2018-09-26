import datetime
import os
import pickle
import subprocess
import sys


from sklearn.base import BaseEstimator


class Model():
    def predict(self, samples):
        """
        Rule based model: predicts class READER if the word READER is present in the string, else predicts class OTHER
        :param samples: Array of text strings
        :return: Array of predictions
        """
        return ['READER' if s.lower().count('reader') else 'OTHER' for s in samples]


model_name = 'model.pkl'
model = Model()
Model.__module__ = 'rule'

# Export the classifier to a file
with open('model.pkl', 'wb') as model_file:
  pickle.dump(model, model_file)

# Upload the saved model file to Cloud Storage
model_path = os.path.join('gs://', 'guillermo-ml-test',
                          datetime.datetime.now().strftime('rule_%Y%m%d_%H%M%S'),
                          model_name)
subprocess.check_call(['gsutil', 'cp', model_name, model_path], stderr=sys.stdout)