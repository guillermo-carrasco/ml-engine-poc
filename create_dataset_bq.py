import os

import numpy as np
import pandas as pd

from google.cloud import bigquery
from google.cloud.bigquery import SchemaField
from sklearn.model_selection import train_test_split


def get_twitter_dataset(overwrite=False):
    """
    Returns a twitter dataset to be written to BigQuery
    :param overwrite: Overwrite existing dataset
    :return: dataset - Pandas Dataframe
    """
    if os.path.exists('data/data.csv') and not overwrite:
        return pd.read_csv('data/data.csv')

    data = pd.read_csv('data/train.csv', encoding='latin-1')
    x = data['SentimentText']
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

    # Build some simple features
    data['language'] = 'EN'
    data['twit_length'] = data['SentimentText'].apply(len)
    data['Set'] = np.nan
    data.loc[X_train.index, 'Set'] = 'TRAIN'
    data.loc[X_test.index, 'Set'] = 'TEST'

    data.to_csv('data/data.csv')

    return data


def create_bigquery_dataset(ds_name, table_name, data):
    """
    Creates a dataset for the twits and fills it up with twit data, if it does not exist yet
    :param ds_name: dataset name in BigQuery
    :param table_name: talbe name for twit data within the dataset
    :param data: data to be uploaded
    :return:
    """
    # Define BigQuery schema
    schema = [
        SchemaField('Sentiment', 'INT64', mode='required'),
        SchemaField('SentimentText', 'STRING', mode='required'),
        SchemaField('language', 'STRING', mode='required'),
        SchemaField('twit_length', 'INT64', mode='required'),
        SchemaField('Set', 'STRING', mode='required')
    ]
    client = bigquery.Client()

    # Get or create the dataset
    dataset_ref = client.dataset(ds_name)
    dataset = bigquery.Dataset(dataset_ref)
    try:
        dataset = client.get_dataset(dataset)
        print('Dataset present')
    except:
        print('Dataset not found, creating...')
        dataset.location = 'EU'
        dataset = client.create_dataset(dataset)

    # Create table if it does not exist
    tables = list(client.list_tables(dataset_ref))
    table_ref = dataset_ref.table(table_name)
    table = bigquery.Table(table_ref, schema=schema)
    if table_name in [t.table_id for t in tables]:
        print('Table of twits already created')
        table = client.get_table(table_ref)
    else:
        print('Table twits not found, creating...')
        table = client.create_table(table)

    # Fill in table data
    print('Filling table twits...')
    rows = [tuple(r[1:]) for r in data.values]

    def _batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield ndx, iterable[ndx:min(ndx + n, l)]

    # Iterate in batches to not exceed payload size limit
    for batch_n, batch in _batch(rows, n=1000):
        print('Uploading batch {}'.format(str(batch_n)))
        errors = client.insert_rows(table, batch)
        if errors:
            print('There were errors copying batch {}: {}'.format(str(batch_n, '\n'.join(errors))))


if __name__ == '__main__':
    data = get_twitter_dataset()
    create_bigquery_dataset('twitter_data', 'twits', data)