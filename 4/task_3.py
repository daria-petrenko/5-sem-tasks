import numpy as np


class BatchGenerator:
    def __init__(self, list_of_sequences, batch_size, shuffle=False):
        self.batch_size = batch_size
        if(shuffle):
            for seq in list_of_sequences:
                if(type(seq) == list):
                    seq = np.array(seq)
                np.random.shuffle(seq)
        self.list_of_seq = list_of_sequences

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if(self.ind >= np.size(self.list_of_seq[0])):
            raise StopIteration
        batch_list = list()
        for seq in self.list_of_seq:
            batch_list.append(seq[self.ind: self.ind + self.batch_size])
        self.ind += self.batch_size
        return batch_list
