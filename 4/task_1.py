import numpy as np


def encode_rle(x):
    y = np.hstack((np.ones(1), x[:np.size(x) - 1]))
    first_positions = x != y
    first_positions[0] = True
    indexes_1 = np.arange(np.size(x))[first_positions]
    indexes_2 = np.hstack((indexes_1[1:], np.array([np.size(x)])))
    return x[first_positions], indexes_2 - indexes_1

def decode_rle(val_arr, num_arr):
    res_arr = np.zeros(np.sum(num_arr))
    ind = 0
    for i in range(np.size(num_arr)):
        for j in range(num_arr[i]):
            res_arr[ind] = val_arr[i]
            ind += 1
    return res_arr.astype('int')
            


class RleSequence:
    def __init__(self, input_sequence):
        self.values, self.numbers = encode_rle(input_sequence)

    def __iter__(self):
        self.val_ind = 0  # current index in values and numbers vectors
        self.curr_num = 0  # number of values of this type already written
        return self

    def __next__(self):
        if(self.curr_num >= self.numbers[self.val_ind]):
            self.val_ind += 1
            self.curr_num = 0
        if(self.val_ind >= np.size(self.values)):
            raise StopIteration
        else:
            self.curr_num += 1
            return self.values[self.val_ind]

    def __getitem__(self, ind):
        if isinstance(ind, slice): 
            start = ind.start
            stop = ind.stop
            step = ind.step
            if(step is None):
                step = 1
            if(stop is None):
                stop = np.sum(self.numbers) 
            if(start is None):
                start = 0
            return decode_rle(self.values, self.numbers)[start:stop:step]
        else:
            if ind < 0:
                ind = np.sum(self.numbers) + ind
            if(ind < 0 or ind >= np.sum(self.numbers)):
                raise IndexError
            sum = 0
            i = 0
            while(i < np.size(self.numbers)):
                sum += self.numbers[i]
                if(ind < sum):
                    return self.values[i]
                i += 1

    def __contains__(self, elem):
        for val in self.values:
            if(elem == val):
                return True
