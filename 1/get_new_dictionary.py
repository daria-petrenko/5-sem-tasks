def get_new_dictionary(input_dict_name, output_dict_name):
    f_in = open(input_dict_name, 'r')
    f_out = open(output_dict_name, 'w')
    d = {}
    line = f_in.readline()
    number = int(line)
    for i in range(number):
        line = f_in.readline()
        end = line.find(' ')
        human_word = line[:end]
        if line[len(line) - 1] == '\n':
            line = line[end + 3:-1]
        else:
            line = line[end + 3:]
        lst = line.split(', ')
        for dragon_word in lst:
            if dragon_word in d:
                d[dragon_word].append(human_word)
            else:
                d[dragon_word] = [human_word]
    f_out.write(str(len(d)) + '\n')
    keys = sorted(d)
    for key in keys:
        f_out.write(key + ' - ' + ', '.join(sorted(d[key])) + '\n')
    f_in.close()
    f_out.close()
