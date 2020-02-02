from dataextractor import get_data, MergeData
import pandas as pd
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    download_data = False
    merge = False
    clean = True
    standardize = True
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
        # print(train.isna().sum())
        rows_before = train.shape[0]
        print('Train set before cleaning:', rows_before)
        train = train[(train.HOME_TEAM_ID.notnull()) & (train.VISITOR_TEAM_ID.notnull())]

        for col in train.columns:
            nullsInCol = train[col].isna().sum()
            print('nulls in col ', col, nullsInCol)
            if nullsInCol > 0:
                media_col = train[[col]].mean()
                train[col] = train[[col]].fillna(media_col)

        # train = train.dropna()
        # print(train.isna().sum())
        rows_after = train.shape[0]
        print('Train set after cleaning:', rows_after)
        print('Deleted rows:', rows_before - rows_after)

        train['HOME_WL'] = train['HOME_WL'].map({'L': 0, 'W': 1})
        train['VISITOR_WL'] = train['VISITOR_WL'].map({'L': 0, 'W': 1})

        train = train.drop(
            ['MATCHUP', 'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'],
            axis=1)

        # print(train.loc[1:10, ['GAME_ID', 'HOME_WL', 'VISITOR_WL']])
        print('Saving cleaned train dataset:', rows_before - rows_after)
        train.to_csv(r'.\datasets\cleanTrain.csv')

        print('DONE saving')

    if standardize:
        cleanTrain = pd.read_csv('./datasets/cleanTrain.csv', index_col=0, header=0, low_memory=False)

        from sklearn.preprocessing import StandardScaler

        notStandardize = ['GAME_ID', 'SEASON_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']
        for col in cleanTrain.columns:
            if col not in notStandardize:
                cleanTrain[col] = StandardScaler().fit_transform(cleanTrain[[col]])
                print('nulls in col ', col, cleanTrain[col].isna().sum())

    if train_model:
        from sklearn.model_selection import train_test_split

        training_set, validation_set = train_test_split(cleanTrain, test_size=0.2, random_state=21)

        # classifying the predictors and target variables as X and Y
        # X_train = training_set.iloc[:, 0:-1].values
        # Y_train = training_set.iloc[:, -1].values
        # X_val = validation_set.iloc[:, 0:-1].values
        # y_val = validation_set.iloc[:, -1].values

        y_val = validation_set.loc[:, ['HOME_WL', 'VISITOR_WL']].values
        X_val = validation_set.drop(['GAME_DATE', 'HOME_WL', 'VISITOR_WL'], axis=1).values
        Y_train = training_set.loc[:, ['HOME_WL', 'VISITOR_WL']].values
        X_train = training_set.drop(['GAME_DATE', 'HOME_WL', 'VISITOR_WL'], axis=1).values


        def accuracy(confusion_matrix):
            diagonal_sum = confusion_matrix.trace()
            sum_of_all_elements = confusion_matrix.sum()
            return diagonal_sum / sum_of_all_elements


        # Importing MLPClassifier
        from sklearn.neural_network import MLPClassifier

        # Initializing the MLPClassifier
        classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                                   random_state=1)

        # Fitting the training data to the network
        classifier.fit(X_train, Y_train)

        # Predicting y for X_val
        y_pred = classifier.predict(X_val)

        # Importing Confusion Matrix
        from sklearn.metrics import confusion_matrix

        # Comparing the predictions against the actual observations in y_val
        cm = confusion_matrix(y_pred, y_val)

        # Printing the accuracy
        print("Accuracy of MLPClassifier : '", accuracy(cm))

        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # clf.fit(X, y)
