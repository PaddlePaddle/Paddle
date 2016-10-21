from paddle.trainer.PyDataProvider2 import *

# id of the word not in dictionary
UNK_IDX = 0

# initializer is called by the framework during initialization.
# It allows the user to describe the data types and setup the
# necessary data structure for later use.
# `settings` is an object. initializer need to properly fill settings.input_types.
# initializer can also store other data structures needed to be used at process().
# In this example, dictionary is stored in settings.
# `dictionay` and `kwargs` are arguments passed from trainer_config.lr.py
def initializer(settings, utt_pdf, num_senone, **kwargs):
    # Put the word dictionary into settings
    settings.utt_pdf = utt_pdf

    # setting.input_types specifies what the data types the data provider
    # generates.
    settings.input_types = [
        dense_float_vector(484), # FIXME: read the input dim from somewhere
        integer_value(num_senone)]

# Delaring a data provider. It has an initializer 'data_initialzer'.
# It will cache the generated data of the first pass in memory, so that
# during later pass, no on-the-fly data generation will be needed.
# `setting` is the same object used by initializer()
# `file_name` is the name of a file listed train_list or test_list file given
# to define_py_data_sources2(). See trainer_config.lr.py.
@provider(init_hook=initializer, cache=None)
def process(settings, file_name):
    # Open the input data file.
    utt_pdf = settings.utt_pdf
    with open(file_name, 'r') as fin:
        while not f.eof():
            key = ''
            while True:
                buf = fin.read(1)
                if buf == ' ':
                    break
                key = key + buf
            pdf = utt_pdf[key] # FIXME: catch the KeyError
            buf = fin.read(15)
            assert buf[:6] == const_tag and buf[10] == chr(0x04)
            num_frame = struct.unpack('<i', buf[6:10])[0]
            dim = struct.unpack('<i', buf[11:15])[0]
            buf = fin.read(4 * dim * num_frame) # 4 is size of float
            buf = buf[:dim] * 4 + buf + buf[-dim:] * 4 # Pad 4 frames to head and tail
            for row in range(num_frame):
                feat = struct.unpack('<%df' % (dim*11), buf[row*dim*4 : (row+11)*dim*4]) # 4 is size of float
                yield feat, pdf[row]


