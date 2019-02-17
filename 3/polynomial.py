class Polynomial:

    def __init__(self, *args):
        self.arg_list = args

    def __call__(self, x):
        sum = 0
        curr_x = 1
        for arg in self.arg_list:
            sum += curr_x * arg
            curr_x *= x
        return sum
