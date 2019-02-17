def find_max_substring_occurrence(string):
    for k in reversed(range(len(string) + 1)):  # choosing k
        if len(string) % k == 0:  # if string can be divided into k parts
            curr_len = len(string) // k   # length of substring
            if string == string[0:curr_len] * k:
                return k
