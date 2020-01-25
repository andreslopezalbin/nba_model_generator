class Game:
    def __init__(self, data, local, visitor):
        self.id = data[2]
        self.date = data[0]
        self.local = local
        self.visitor = visitor
        self.arena = data[15]

    def __str__(self):
        return f'Game: {self.id}, Date: {self.date}, Visitor Team: {self.visitor.full_name}, ' \
               f'Local Team: {self.local.full_name}, Arena: {self.arena}'
