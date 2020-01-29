from dataextractor import get_data, MergeData
import pandas as pd

if __name__ == '__main__':
    download_data = False
    merge = False
    # Download Data
    if download_data:
        data = get_data()
    else:
        data = pd.read_csv('./datasets/dataset.csv', index_col=None, header=0)

    # Merge Data
    if merge:
        MergeData(data)

    # Clean Data
    train = pd.read_csv('./datasets/train.csv', index_col=0, header=0)
    print(train[0:5])
    # print(train.isna().sum())
    rows_before = train.shape[0]
    print('Train set before cleaning:', rows_before)
    train = train[(train.HOME_TEAM_ID.notnull()) & (train.VISITOR_TEAM_ID.notnull())]
    # print(train.isna().sum())
    rows_after = train.shape[0]
    print('Train set after cleaning:', rows_after)
    print('Deleted rows:', rows_before - rows_after)
