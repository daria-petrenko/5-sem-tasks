class CooSparseMatrix:

    def __init__(self, ijx_list, shape):
        self._shape = shape
        self.dict = dict()
        for elem in ijx_list:
            if(type(elem[0]) == float) or (type(elem[1]) == float):
                raise TypeError
            if((int(elem[0]) >= self._shape[0]) or (int(elem[1]) >= self._shape[1])):
                raise TypeError
            if((int(elem[0]), int(elem[1])) in self.dict):
                raise TypeError
            elif(int(elem[2]) != 0):
                self.dict[(int(elem[0]), int(elem[1]))] = elem[2]

    def __getitem__(self, args):
        if(type(args) != tuple):
            if(type(args) == float):
                raise TypeError
            if(int(args) >= self._shape[0]):
                raise TypeError
            curr_ijx_list = list()
            for j in range(self._shape[1]):
                if((int(args), j) in self.dict):
                    curr_ijx_list.append((int(args), j, self.dict[(int(args), j)]))
            return CooSparseMatrix(curr_ijx_list, (1, self._shape[1]))
        else:
            if(type(args[0]) == float) or (type(args[1]) == float):
                raise TypeError
            if((int(args[0]) >= self._shape[0]) or (int(args[1]) >= self._shape[1])):
                raise TypeError
            if((int(args[0]), int(args[1])) in self.dict):
                return self.dict[(int(args[0]), int(args[1]))]
            else:
                return 0

    def __setitem__(self, args, value):
        if(type(args[0]) == float) or (type(args[1]) == float):
                raise TypeError
        if((int(args[0]) >= self._shape[0]) or (int(args[1]) >= self._shape[1])):
                raise TypeError
        if(int(value) != 0):
            self.dict[(int(args[0]), int(args[1]))] = value
        elif((int(args[0]), int(args[1])) in self.dict):
            del(self.dict[int(args[0]), int(args[1])])

    def __add__(self, other):
        if((self._shape[0] != other._shape[0]) or (self._shape[1] != other._shape[1])):
            raise TypeError
        curr_ijx_list = list()
        for key in self.dict:
            curr_value = self.dict[key]
            if(key in other.dict):
                curr_value += other.dict[key]
            if(curr_value != 0):
                curr_ijx_list.append((key[0], key[1], curr_value))
        for key in other.dict:
            if(not(key in self.dict)):
                curr_ijx_list.append((key[0], key[1], other.dict[key]))
        return CooSparseMatrix(curr_ijx_list, self._shape)

    def __sub__(self, other):
        if((self._shape[0] != other._shape[0]) or (self._shape[1] != other._shape[1])):
            raise TypeError
        curr_ijx_list = list()
        for key in self.dict:
            curr_value = self.dict[key]
            if(key in other.dict):
                curr_value -= other.dict[key]
            if(curr_value != 0):
                curr_ijx_list.append((key[0], key[1], curr_value))
        for key in other.dict:
            if(not(key in self.dict)):
                curr_ijx_list.append((key[0], key[1], -1 * other.dict[key]))
        return CooSparseMatrix(curr_ijx_list, self._shape)

    def __mul__(self, value):
        curr_ijx_list = list()
        for key in self.dict:
            curr_ijx_list.append((key[0], key[1], value * self.dict[key]))
        return CooSparseMatrix(curr_ijx_list, self._shape)

    def __rmul__(self, value):
        return self * value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if(type(value) != tuple):
            raise TypeError
        if(len(value) != 2):
            raise TypeError
        if(value[0] * value[1] != self._shape[0] * self._shape[1]):
            raise TypeError
        if((type(value[0]) == float) or (type(value[1]) == float)):
            raise TypeError
        new_dict = dict()
        for key in self.dict:
            pos = self._shape[1] * key[0] + (key[1] + 1)
            new_i = (pos - 1) // value[1]
            new_j = (pos - 1) % value[1]
            new_dict[(new_i, new_j)] = self.dict[key]
        self._shape = value
        self.dict = new_dict

    @property
    def T(self):
        new_ijx_list = list()
        for key in self.dict:
            new_ijx_list.append((key[1], key[0], self.dict[key]))
        return CooSparseMatrix(new_ijx_list, (self._shape[1], self._shape[0]))

    @T.setter
    def T(self, value):
        raise AttributeError
