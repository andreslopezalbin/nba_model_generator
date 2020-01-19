from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

nba_teams = teams.get_teams()
# Select the dictionary for the Celtics, which contains their team ID
#celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]
#celtics_id = celtics['id']
#print('celtics ID: ', celtics_id)

for team in nba_teams:
	print(team['abbreviation'], team['id'])
	gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
	#gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'],date_from_nullable='12/12/2019')
	games = gamefinder.get_data_frames()[0]
	games.to_csv(r'.\datasets\\'+team['abbreviation']+'.csv');