import json
import pymongo
import pickle
import os
import boto3
import io
import datetime

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

db_url = os.environ['db_url']


# Obtencion de los datos de entrenamiento desde el Bucket S3
def get_dataset():
    print("Getting S3 object...")
    s3 = boto3.client('s3')
    data = s3.get_object(Bucket='nba-datasets-bucket', Key='train.csv')
    return pd.read_csv(io.BytesIO(data['Body'].read()), index_col=0)


def train(event, context):

    # Lectura del dataset
    train_df = get_dataset()

    # Calculo de la clase a predecir
    train_df['PLUS_MINUS'] = train_df['HOME_PTS'] - train_df['VISITOR_PTS']

    # Se eliminan los registros que no dispongan de los datos de los dos equipos
    train_df = train_df[(train_df.HOME_TEAM_ID.notnull()) & (train_df.VISITOR_TEAM_ID.notnull())]

    # Limpieza de literales y parámetros que pueden interferir con los resultados del experimento.
    train_df = train_df.drop(
        ['GAME_ID', 'SEASON_ID', 'HOME_WL', 'VISITOR_WL', 'HOME_PTS', 'VISITOR_PTS', 'GAME_DATE', 'MATCHUP',
         'HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME',  'HOME_TEAM_ID',
         'VISITOR_TEAM_ID'], axis=1)

    # En caso de que una estadística este vacía, esta se rellena con la media del dataset.
    for col in train_df.columns:
        nullsInCol = train_df[col].isna().sum()
        if nullsInCol > 0:
            media_col = train_df[[col]].mean()
            train_df[col] = train_df[[col]].fillna(media_col)

    # Se recorre cada una de las columnas escalando sus valores.
    not_standardize = ['PLUS_MINUS']
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

    home_team = event['home']
    visitor_team = event['visitor']

    df = get_dataset()
    df = df.drop(
        ['GAME_ID', 'SEASON_ID','HOME_WL', 'VISITOR_WL', 'HOME_PTS', 'VISITOR_PTS', 'MATCHUP', 'HOME_TEAM_NAME',
         'VISITOR_TEAM_NAME', 'PLUS_MINUS', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], axis=1)

    home_games = df.loc[df['HOME_TEAM_ABBREVIATION'] == home_team ]
    home_games = home_games.sort_values(by="GAME_DATE", ascending=False)[:10]

    visitor_games = df.loc[df['VISITOR_TEAM_ABBREVIATION'] == visitor_team]
    visitor_games = visitor_games.sort_values(by="GAME_DATE", ascending=False)[:10]

    means = {}
    for col in home_games.columns:
        if 'HOME_' in col and col != 'HOME_TEAM_ABBREVIATION' and col != 'GAME_DATE':
            means[col] = home_games[col].mean()

    for col in visitor_games.columns:
        if 'VISITOR_' in col and col != 'VISITOR_TEAM_ABBREVIATION' and col != 'GAME_DATE':
            means[col] = visitor_games[col].mean()

    to_predict = pd.DataFrame(columns=df.columns)
    to_predict = to_predict.append(means, ignore_index=True)
    to_predict = to_predict.drop(['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION', 'GAME_DATE'], axis=1)

    print("connecting to db")
    client = pymongo.MongoClient(db_url)
    nba_models = client.nbaDB.models
    active_model = nba_models.find_one({"active": True})

    print("loading regresor")
    active_regresor = pickle.loads(active_model['model'])

    print("getting prediction")
    prediction = active_regresor.predict(to_predict)
    print("prediction " + ' ,'.join([str(elem) for elem in prediction]))
    to_save = {
        "data": to_predict.to_json(orient="index"),
        "query": home_team + ' - ' + visitor_team,
        "prediction": ' ,'.join([str(elem) for elem in prediction]),
        "model": active_model['_id'],
        "date": datetime.datetime.utcnow(),
    }

    nba_predictions = client.nbaDB.predictions
    nba_predictions.insert_one(to_save)

    body = {
        "message": 'prediction: ' + to_save['prediction'],
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
