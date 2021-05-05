from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd


def get_data():
    # Se obtienen todos lo equipos de la NBA
    nba_teams = teams.get_teams()
    dataframes_list = []
    for team in nba_teams:
        # Se hace una petición a la API para obtener los partidos de cada equipo
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
        games = gamefinder.get_data_frames()[0]
        # Se almacenan los partidos de cada equipo por separado
        games.to_csv(r'.\datasets\by_team\\' + team['abbreviation'] + '.csv')
        dataframes_list.append(games)
    # Se almacenan los datos de todos los partidos en un mismo dataset
    games_dataframe = pd.concat(dataframes_list, axis=0, ignore_index=True)
    games_dataframe.to_csv(r'.\datasets\dataset.csv')

    return games_dataframe


class MergeData:

    def __init__(self, dataframe):
        self.missing_games = 0
        self.games_dataframe = dataframe
        self.merge()

    def merge(self):
        games = []
        # Se obtienen los dos registros de los partidos por id y se llama a la funcion join_games
        for gameId in self.games_dataframe.GAME_ID.unique():
            # print(gameId, ' : ', frame.loc[frame['GAME_ID'] == gameId])
            games_by_id = self.games_dataframe.loc[self.games_dataframe['GAME_ID'] == gameId]
            games.append(self.join_games(games_by_id))

        print('Missing games: ', self.missing_games)
        print('Saving train dataset, it can take a little bit, be patient :)')
        train = pd.concat(games, axis=0, ignore_index=True)
        train.to_csv(r'.\datasets\train.csv')
        print('Number of games and attrs in train set: ', train.shape)

    def join_games(self, games):
        games = games.reset_index(drop=True)
        missing_game = None
        try:
            games.loc[1]
        except KeyError:
            # Existen partidos con un unico registro ya que uno de los dos equipos no es de la NBA debido a partido
            # de entrenamiento o exibición
            print('Missing one game for: ', games.loc[0].GAME_ID, ', Game: ', games.loc[0].MATCHUP, ', Date: ',
                  games.loc[0].GAME_DATE)
            self.missing_games += 1
            missing_game = {'TEAM_ID': None}
        # @ significa "AT" y que el registro pertenece al equipo visitante
        if '@' in games.loc[0].MATCHUP:
            visitor = games.loc[0]
            home = games.loc[1] if not missing_game else missing_game
        else:
            visitor = games.loc[1] if not missing_game else missing_game
            home = games.loc[0]

        # Cabecera de los datos devueltos por la nbaAPI
        attrs_headers = ['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
                         'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
                         'TOV', 'PF']

        attrs_list = {'GAME_ID': [games.loc[0].GAME_ID],
                      'SEASON_ID': [games.loc[0].SEASON_ID],
                      'MATCHUP': [games.loc[0].MATCHUP],
                      'GAME_DATE': [games.loc[0].GAME_DATE],
                      }
        # Se llama a cada columna del dataset en funcion de si el dato pertenece al visitante o al local
        for attr in attrs_headers:
            attrs_list['HOME_' + attr] = home.get(attr, None)
            attrs_list['VISITOR_' + attr] = visitor.get(attr, None)

        game_df = pd.DataFrame(attrs_list)

        return game_df
