def find_word_in_circle(circle, word):
    word_len = len(word)
    if len(circle) == 0:
        return -1
    str = circle * (word_len // len(circle) + 2)  # make string out of circle
    for i in range(len(circle)):  # if 1
        if(str[i:i + word_len] == word):
            return i, 1
    for i in range(len(circle)):  # if -1
        if(str[len(str) - 1 - i:len(str) - 1 - i - word_len:-1] == word):
            return len(circle) - 1 - i, -1
    return -1
