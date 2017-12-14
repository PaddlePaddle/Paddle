DICT_DIM = 3000


@provider(input_types=[integer_sequence(DICT_DIM), integer_value(DICT_DIM)])
def process(settings, filename):
    with open(filename) as f:
        # yield word ids to predict inner word id
        # such as [28, 29, 10, 4], 4
        # It means the sentance is  28, 29, 4, 10, 4.
        yield read_next_from_file(f)
