import glob
from nba_api.stats.endpoints import scoreboardv2
from models.game import Game
from models.team import Team

    # Select the dictionary for the Celtics, which contains their team ID
    # celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]

    # gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'], date_from_nullable='12/12/2019')


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

    # path = r'C:\Users\admin\Desktop\master\MLE\proyecto\MLE\datasets'
    # all_files = glob.glob(path + "/*.csv")

    # li = []

    # for filename in all_files:
    #    if filename != 'C:\\Users\\admin\\Desktop\\master\\MLE\\proyecto\\MLE\\datasets\\train.csv':
    #        # df = pd.read_csv(filename, index_col=None, header=0)
    #        df = pd.read_csv(filename)
    #        # print(filename, df[:5])
    #        li.append(df)

    # frame = pd.concat(li, axis=0, ignore_index=True)