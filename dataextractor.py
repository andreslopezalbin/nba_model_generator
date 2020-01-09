from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

nba_teams = teams.get_teams()
# Select the dictionary for the Celtics, which contains their team ID
celtics = [team for team in nba_teams if team['abbreviation'] == 'BOS'][0]
celtics_id = celtics['id']
print('celtics ID: ', celtics_id)

#gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'],date_from_nullable='12/12/2019')
#games = gamefinder.get_data_frames()[0]
#print(games)

#games.to_csv(r'C:\Users\docencia\Desktop\ml\byDate.csv');

for team in nba_teams:
	print(team['abbreviation'], team['id'])
	gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'],date_from_nullable='12/12/2019')
	games = gamefinder.get_data_frames()[0]
	games.to_csv(r'C:\Users\docencia\Desktop\ml\proyecto\\'+team['abbreviation']+'.csv');

# Query for games where the Celtics were playing
#gamefinder = leaguegamefinder.LeagueGameFinder()
# The first DataFrame of those returned is what we want.
#games = gamefinder.get_data_frames()[0]
#print(games.head())
#games.to_csv(r'C:\Users\docencia\Desktop\nba\dataset.csv')
