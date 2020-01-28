from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd


def get_data():
    nba_teams = teams.get_teams()
    dataframes_list = []
    for team in nba_teams:
        print(team['abbreviation'], team['id'])
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
        games = gamefinder.get_data_frames()[0]
        games.to_csv(r'.\datasets\by_team\\' + team['abbreviation'] + '.csv')
        dataframes_list.append(games)

    games_dataframe = pd.concat(dataframes_list, axis=0, ignore_index=True)
    games_dataframe.to_csv(r'.\datasets\dataset.csv')

    return games_dataframe


class MergeData:

    def __init__(self, dataframe):
        self.missing_games = 0
        self.games_dataframe = dataframe
        self.merge()

    def join_games(self, games):
        game_df = pd.DataFrame()
        games = games.reset_index(drop=True)

        missing_game = None
        try:
            games.loc[1]
        except KeyError:
            print('Missing one game for: ', games.loc[0].GAME_ID, ', Game: ', games.loc[0].MATCHUP, ', Date: ',
                  games.loc[0].GAME_DATE)
            self.missing_games += 1
            missing_game = {'TEAM_ID': None}

        if '@' in games.loc[0].MATCHUP:
            visitor = games.loc[0]
            home = games.loc[1] if not missing_game else missing_game
        else:
            visitor = games.loc[1] if not missing_game else missing_game
            home = games.loc[0]

        game_df = pd.DataFrame({'GAME_ID': [games.loc[0].GAME_ID],
                                'MATCHUP': [games.loc[0].MATCHUP],
                                'HOME_TEAM': home.get('TEAM_ID', None),
                                'VISITOR_TEAM': visitor.get('TEAM_ID', None)
                                })
        return game_df

    def merge(self):
        games = []
        for gameId in self.games_dataframe.GAME_ID.unique():
            # print(gameId, ' : ', frame.loc[frame['GAME_ID'] == gameId])
            games_by_id = self.games_dataframe.loc[self.games_dataframe['GAME_ID'] == gameId]
            games.append(self.join_games(games_by_id))

        print('Missing games: ', self.missing_games)
        print('saving train dataset')
        train = pd.concat(games, axis=0, ignore_index=True)
        train.to_csv(r'.\datasets\train.csv')
        print('Number of games and attrs in train set: ', train.shape)
