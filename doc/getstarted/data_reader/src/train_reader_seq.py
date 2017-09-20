def train_reader(file_name, word_dict):
    UNK_ID = word_dict['<unk>']

    def reader():
        with open(file_name) as f:
            # read each line of file
            for line in f:
                # get label and sentence
                label, sentence = line.split(';')
                # convert word string to word id
                # the word not in dictionary 
                # will be replaced with '<unk>'.
                word_ids = [word_dict.get(w, UNK_ID) for w in sentence.split()]
                # give data to paddle.
                yield word_ids, int(label)

    return reader
