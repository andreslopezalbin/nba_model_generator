from dataextractor import get_data, MergeData
import pandas as pd

if __name__ == '__main__':
    # Download Data
    download_data = True
    if download_data:
        data = get_data()
    else:
        data = pd.read_csv('./datasets/dataset.csv', index_col=None, header=0)

    # Merge Data
    MergeData(data)
    # Clean Data
