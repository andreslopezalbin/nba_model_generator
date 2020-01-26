from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2
from models.game import Game
from models.team import Team

nba_teams = teams.get_teams()

# Select the dictionary for the Celtics, which contains their team ID
# celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]

# for team in nba_teams:
#     print(team['abbreviation'], team['id'])
#     gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
#     # gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'], date_from_nullable='12/12/2019')
#     games = gamefinder.get_data_frames()[0]
#     games.to_csv(r'.\datasets\\' + team['abbreviation'] + '.csv');


# def get_team(team_id):
#     return next((team for team in nba_teams if team["id"] == team_id), None)
#
#
# for index in range(0, 3):
#     scoreboard = scoreboardv2.ScoreboardV2(day_offset=index, game_date='2020/01/26', league_id='00')
#     for data in scoreboard.data_sets[0].data['data']:
#         local = Team(get_team(data[6]))
#         visitor = Team(get_team(data[7]))
#
#         game = Game(data, local, visitor)
#         print(game.__str__())
#


import pandas as pd
import glob

path = r'C:\Users\admin\Desktop\master\MLE\proyecto\MLE\datasets'  # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    if filename != 'C:\\Users\\admin\\Desktop\\master\\MLE\\proyecto\\MLE\\datasets\\raw.csv':
        # df = pd.read_csv(filename, index_col=None, header=0)
        df = pd.read_csv(filename)
        # print(filename, df[:5])
        li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv(r'.\datasets\raw.csv')


# print(frame.info())
# print("GamesIDs: ", len(frame.GAME_ID.unique()))

def get_game(games):
    game_df = pd.DataFrame(columns=['', '', ])
    games = games.reset_index(drop=True)
    if '@' in games.loc[0].MATCHUP:
        visitor = games.loc[0]
        home = games.loc[1]
    else:
        visitor = games.loc[1]
        home = games.loc[0]

    game_df = pd.DataFrame({'GAME_ID': [home.GAME_ID],
                            'MATCHUP': [home.MATCHUP],
                            'HOME_TEAM': [home.TEAM_ID],
                            'VISITOR_TEAM': [visitor.TEAM_ID]
                            })

    print(game_df)
    return game_df


for gameId in frame.GAME_ID.unique():
    # print(gameId, ' : ', frame.loc[frame['GAME_ID'] == gameId])
    games = frame.loc[frame['GAME_ID'] == gameId]
    games
    game = []
    get_game(games)
    print(games)
