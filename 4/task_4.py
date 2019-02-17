class WordContextGenerator:
    def __init__(self, words, window_size):
        self.words = words
        self.window_size = window_size

    def __iter__(self):
        self.first = 0
        self.second = 0
        return self

    def __next__(self):
        if(self.first == len(self.words) - 1 and
           self.second == len(self.words) - 1):
            raise StopIteration
        else:
            if(self.first == self.second):
                self.second += 1
            if(self.second >= len(self.words) or
                    abs(self.second - self.first) > self.window_size):
                self.first += 1
                self.second = max(self.first - self.window_size + 1, 1)
            else:
                self.second += 1
            return (self.words[self.first], self.words[self.second - 1])
