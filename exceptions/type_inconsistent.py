

class TypeInconsistent(Exception):
    def __init__(self, info=''):
        Exception.__init__(self)
        self.info = info
