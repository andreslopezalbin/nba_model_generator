from dataextractor import get_data, MergeData
import pandas as pd

from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

    download_data = False
    merge = False
    clean = False
    train_model = True
    predict = True

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

        train['PLUS_MINUS'] = train['HOME_PTS'] - train['VISITOR_PTS']

        train.to_csv(r'.\datasets\train.csv')

        print('DONE saving')

    if train_model:
        train = pd.read_csv('./datasets/train.csv', index_col=0, header=0)

        train = train[(train.HOME_TEAM_ID.notnull()) & (train.VISITOR_TEAM_ID.notnull())]
        train = train.drop(
            ['HOME_WL', 'VISITOR_WL', 'HOME_PTS', 'VISITOR_PTS', 'GAME_DATE', 'MATCHUP', 'HOME_TEAM_ABBREVIATION',
             'VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'], axis=1)

        train = train[(train.HOME_TEAM_ID.notnull()) & (train.VISITOR_TEAM_ID.notnull())]
        train = train.astype({'HOME_TEAM_ID': 'int64', 'VISITOR_TEAM_ID': 'int64'})

        # train.info()

        # Clean null values
        for col in train.columns:
            nullsInCol = train[col].isna().sum()
            # print('nulls in col ', col, nullsInCol)
            if nullsInCol > 0:
                media_col = train[[col]].mean()
                train[col] = train[[col]].fillna(media_col)

        y = train['PLUS_MINUS']
        X = train.drop(['PLUS_MINUS'], axis=1)

        from sklearn.preprocessing import StandardScaler

        notStandardize = ['GAME_ID', 'SEASON_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']
        for col in X.columns:
            if col not in notStandardize:
                X[col] = StandardScaler().fit_transform(X[[col]])

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn import metrics

        regresor = LinearRegression()
        regresor.fit(X, y)

        # Predicción cruzada
        # predicciones = regresor.predict(X[:10])
        # print(predicciones)

        # Evaluación cruzada
        # scores = cross_val_score(regresor, X, y, cv=10)
        # print(scores.mean())

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=123)
        # regresor = regresor.fit(X_train, y_train)
        # y_test_pred = regresor.predict(X_test)

        # metrics.r2_score(y_test, y_test_pred)

        score = cross_val_score(regresor, X, y, cv=10, scoring='neg_mean_absolute_error')
        mae = score.mean()
        score = cross_val_score(regresor, X, y, cv=10, scoring='neg_mean_squared_error')
        mse = score.mean()
        score = cross_val_score(regresor, X, y, cv=10, scoring='r2')
        r2 = score.mean()

        print('MAE:', mae)
        print('MSE:', mse)
        print('R2:', r2)

        import pymongo
        import pickle
        import datetime

        pickled_model = pickle.dumps(regresor)
        print("connecting to db")
        client = pymongo.MongoClient(
            "mongodb+srv://admin:b5LlXLx9dAUw7amF@cluster0-k0rto.mongodb.net/nbaDB?retryWrites=true&w=majority")
        nba_models = client.nbaDB.models
        active_model = nba_models.find_one({"active": True})
        active = False
        message = ''
        if active_model is None:
            active = True
            message = 'Active model not found'
            print(message)
        elif active_model['scores']['R2'] < r2:
            active = True

            nba_models.update_one({'_id': active_model['_id']}, {"$set": {"active": False}})
            message = ' Better model found. Scores R2: ' + str(r2) + ' --- MAE: ' + str(mae) + ' --- MSE: ' + str(mse)
            print(message)
        else:
            message = ' Better model NOT found. Active model scores R2: ' + str(active_model['scores']['R2']) \
                      + ' --- MAE: ' + str(active_model['scores']['MAE']) \
                      + ' --- MSE: ' + str(active_model['scores']['MSE']) \
                      + ' ### NEW model scores R2: ' + str(r2) \
                      + ' --- MAE: ' + str(mae) \
                      + ' --- MSE: ' + str(mse)
            print(message)
        model = {
            "model": pickled_model,
            "date": datetime.datetime.utcnow(),
            "active": active,
            "scores": {
                "R2": r2,
                "MAE": mae,
                "MSE": mse
            }
        }
        print('saving model')
        nba_models.insert_one(model)

    if predict:
        import pymongo
        import pickle
        print("connecting to db")
        client = pymongo.MongoClient(
            "mongodb+srv://admin:b5LlXLx9dAUw7amF@cluster0-k0rto.mongodb.net/nbaDB?retryWrites=true&w=majority")

        nba_models = client.nbaDB.models
        active_model = nba_models.find_one({"active": True})
        print("loading regresor")
        active_regresor = pickle.loads(active_model['model'])

        import numpy as np

        columns = ['GAME_ID', 'SEASON_ID', 'MATCHUP', 'GAME_DATE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
                   'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME',
                   'HOME_WL', 'VISITOR_WL', 'HOME_MIN', 'VISITOR_MIN', 'HOME_PTS', 'VISITOR_PTS', 'HOME_FGM',
                   'VISITOR_FGM', 'HOME_FGA', 'VISITOR_FGA', 'HOME_FG_PCT', 'VISITOR_FG_PCT', 'HOME_FG3M',
                   'VISITOR_FG3M', 'HOME_FG3A', 'VISITOR_FG3A', 'HOME_FG3_PCT', 'VISITOR_FG3_PCT', 'HOME_FTM',
                   'VISITOR_FTM', 'HOME_FTA', 'VISITOR_FTA', 'HOME_FT_PCT', 'VISITOR_FT_PCT',
                   'HOME_OREB', 'VISITOR_OREB', 'HOME_DREB', 'VISITOR_DREB', 'HOME_REB', 'VISITOR_REB', 'HOME_AST',
                   'VISITOR_AST', 'HOME_STL', 'VISITOR_STL', 'HOME_BLK', 'VISITOR_BLK', 'HOME_TOV', 'VISITOR_TOV',
                   'HOME_PF', 'VISITOR_PF']

        game = [[111, 22019, 'GSW vs. MIA', '10/02/2020', 1610612744, 1610612748, 'GSW', 'MIA', 'Golden State Warriors',
                 'Miami' 'Heat', 'x', 'x', 245, 249.6, 109.2, 116.6, 38.6, 40, 90, 85.8, 0.4308, 0.469, 12.2, 13.8,
                 35.4, 36.8, 0.3362, 0.3868, 19.8, 22.8, 24, 26.8, 0.8208, 0.8568, 9.2, 6.8, 31.8, 36.4, 41, 43.2, 26.8,
                 25.2, 7, 5.8, 4.4, 4.8, 13.2, 13.2, 21.4, 19.2]]

        df = pd.DataFrame(np.array(game), columns=columns)
        df = df.drop(
            ['HOME_WL', 'VISITOR_WL', 'HOME_PTS', 'VISITOR_PTS', 'GAME_DATE', 'MATCHUP', 'HOME_TEAM_ABBREVIATION',
             'VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'], axis=1)

        print("getting prediction")
        print(active_regresor.predict(df))
