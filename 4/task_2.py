class linearize:
    def __init__(self, x):
        self.object = x
        self.iter = iter(self.object)
        self.rec = None

    def __iter__(self):
        return self

    def __next__(self):
        if(self.rec is not None):
            try:
                elem = self.rec.__next__()
            except StopIteration:
                self.rec = None
            else:
                return elem
        if(self.rec is None):
            try:
                elem = next(self.iter)
                if(type(elem) == str and len(elem) == 1):
                    return elem
                iter(elem)  # check if is iterable
            except TypeError:  # if not iterable, then return element
                return elem
            else:  # if iterable, then iterare recursively
                self.rec = linearize(elem)
                try:
                    elem = self.rec.__next__()
                except StopIteration:
                    self.rec = None
                    self.__next__()
                else:
                    return elem
