class Team:
    def __init__(self, data):
        self.id = data['id']
        self.full_name = data['full_name']
        self.abbreviation = data['abbreviation']
        self.nickname = data['nickname']
        self.city = data['city']
        self.state = data['state']
        self.year_founded = data['year_founded']

    def __str__(self):
        return f'Team: {self.id}, full_name: {self.full_name}, Abbreviation: {self.abbreviation}, ' \
               f'Year Founded: {self.year_founded}'

#  Response example
#   id': 1610612737,
#   'full_name': 'Atlanta Hawks',
#   'abbreviation': 'ATL',
#   'nickname': 'Hawks',
#   'city': 'Atlanta',
#   'state': 'Atlanta',
#   'year_founded': 1949},
