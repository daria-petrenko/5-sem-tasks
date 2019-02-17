def subreverse_sequence(seq):
    parity = (len(seq) + 1) % 2
    t = tuple(seq[len(seq) - 1 - parity::-2] + seq[1::2])
    return t
