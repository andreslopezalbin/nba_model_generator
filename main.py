from dataextractor import get_data, MergeData
import pandas as pd
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    download_data = False
    merge = True
    clean = True
    standardize = False
    train_model = False
    # Download Data
    if download_data:
        data = get_data()
    else:
        data = pd.read_csv('./datasets/dataset.csv', index_col=0, header=0)

    # Merge Data
    if merge:
        MergeData(data)

    # Clean Data
    if clean:
        train = pd.read_csv('./datasets/train.csv', index_col=0, header=0)
        print(train[0:5])
        train['HOME_WL'] = train['HOME_WL'].map({'L': 0, 'W': 1})
        train['VISITOR_WL'] = train['VISITOR_WL'].map({'L': 0, 'W': 1})
        train['PLUS_MINUS'] = train['HOME_PTS'] - train['VISITOR_PTS']

        train.to_csv(r'.\datasets\train.csv')

        print('DONE saving')

    #if train_model:

