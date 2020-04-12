import json
import pymongo
import pickle
import pandas as pd
import numpy as np


def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response


def predict(event, context):
    print("connecting to db")
    client = pymongo.MongoClient(
        "mongodb+srv://admin:nN9nWsRTD98DKfyQ@cluster0-k0rto.mongodb.net/nbaDB?retryWrites=true&w=majority")

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
