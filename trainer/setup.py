from setuptools import setup
setup(name='trainer',
      version='1.0',
      packages=['trainer'],
      install_requires=[
            'google-cloud-bigquery>=0.32.0',
            'pandas-gbq',
            'google-cloud-bigquery'
      ],
      zip_safe=False)