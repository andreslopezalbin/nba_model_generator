import json
import pymongo
import pickle
import os
import boto3
import io
import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

db_url = os.environ['db_url']


def train(event, context):
    # Obtencion de los datos de entrenamiento desde el Bucket S3
    s3 = boto3.client('s3')
    print("Getting S3 object...")
    dataset = s3.get_object(Bucket='nba-datasets-bucket', Key='train.csv')

    # Lectura del dataset
    train_df = pd.read_csv(io.BytesIO(dataset['Body'].read()), index_col=0)

    # Calculo de la clase a predecir
    train_df['PLUS_MINUS'] = train_df['HOME_PTS'] - train_df['VISITOR_PTS']

    # Limpieza de literales y parámetros que pueden interferir con los resultados del experimento.
    train_df = train_df.drop(
        ['HOME_WL', 'VISITOR_WL', 'HOME_PTS', 'VISITOR_PTS', 'GAME_DATE', 'MATCHUP', 'HOME_TEAM_ABBREVIATION',
         'VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'], axis=1)

    # Se eliminan los registros que no dispongan de los datos de los dos equipos
    train_df = train_df[(train_df.HOME_TEAM_ID.notnull()) & (train_df.VISITOR_TEAM_ID.notnull())]
    train_df = train_df.astype({'HOME_TEAM_ID': 'int64', 'VISITOR_TEAM_ID': 'int64'})

    # En caso de que una estadística este vacía, esta se rellena con la media del dataset.
    for col in train_df.columns:
        nullsInCol = train_df[col].isna().sum()
        if nullsInCol > 0:
            media_col = train_df[[col]].mean()
            train_df[col] = train_df[[col]].fillna(media_col)

    # Se recorre cada una de las columnas escalando sus valores.
    not_standardize = ['GAME_ID', 'SEASON_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'PLUS_MINUS']
    for col in train_df.columns:
        if col not in not_standardize:
            train_df[col] = StandardScaler().fit_transform(train_df[[col]])

    y = train_df['PLUS_MINUS']
    X = train_df.drop(['PLUS_MINUS'], axis=1)

    regresor = LinearRegression()
    regresor.fit(X, y)

    score = cross_val_score(regresor, X, y, cv=10, scoring='neg_mean_absolute_error')
    mae = score.mean()
    score = cross_val_score(regresor, X, y, cv=10, scoring='neg_mean_squared_error')
    mse = score.mean()
    score = cross_val_score(regresor, X, y, cv=10, scoring='r2')
    r2 = score.mean()

    print('MAE:', mae, ' -- MSE:', mse, ' -- R2:', r2)

    pickled_model = pickle.dumps(regresor)

    print("connecting to DB")

    client = pymongo.MongoClient(db_url)
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

    print('Saving model on DB')
    nba_models.insert_one(model)

    body = {
        "message": message,
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response


def predict(event, context):
    print("connecting to db")
    client = pymongo.MongoClient(db_url)
    nba_models = client.nbaDB.models
    active_model = nba_models.find_one({"active": True})
    print("loading regresor")
    active_regresor = pickle.loads(active_model['model'])

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
    prediction = active_regresor.predict(df)
    body = {
        "message": "Prediction: " + ' ,'.join([str(elem) for elem in prediction]),
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
