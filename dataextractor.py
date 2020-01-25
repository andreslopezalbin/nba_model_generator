from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2
from models.game import Game
from models.team import Team

nba_teams = teams.get_teams()

# Select the dictionary for the Celtics, which contains their team ID
# celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]

for team in nba_teams:
    print(team['abbreviation'], team['id'])
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
    # gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'], date_from_nullable='12/12/2019')
    games = gamefinder.get_data_frames()[0]
    games.to_csv(r'.\datasets\\' + team['abbreviation'] + '.csv');


def get_team(team_id):
    return next((team for team in nba_teams if team["id"] == team_id), None)


for index in range(0, 3):
    scoreboard = scoreboardv2.ScoreboardV2(day_offset=index, game_date='2020/01/26', league_id='00')
    for data in scoreboard.data_sets[0].data['data']:
        local = Team(get_team(data[6]))
        visitor = Team(get_team(data[7]))

        game = Game(data, local, visitor)
        print(game.__str__())



# DATOS_PD = pd.read_csv('./train-bike.csv')
