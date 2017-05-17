# edit-mode: -*- python -*-

from paddle.trainer_config_helpers import *

trn = '/paddle/data/train.list'
tst = '/paddle/data/test.list'
label = '/paddle/data/100k_utt.pdf'
process = 'process'

utt_pdf = dict()
with open(label) as f:
    for line in f:
        parts = line.split()
        utt_pdf[parts[0]] = [int(s) for s in parts[1:]]


define_py_data_sources2(train_list=trn,
                        test_list=tst,
                        module="dataprovider_kaldi",
                        obj=process,
                        args={"utt_pdf" : utt_pdf, "num_seonoe" : 3513})

batch_size = 128
settings(
    batch_size=batch_size,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25
)

# Define the data for text features. The size of the data layer is the number
# of words in the dictionary.
data = data_layer(name="spliced_mfcc", size=484)
hidden1 = fc_layer(input=data, size=1024, act=SigmoidActivation())
hidden2 = fc_layer(input=hidden1, size=1024, act=SigmoidActivation())
output = fc_layer(input=hidden2, size=3513, act=SoftmaxActivation())

# For training, we need label and cost

# define the category id for each example.
# The size of the data layer is the number of labels.
label = data_layer(name="label", size=3513)

# Define cross-entropy classification loss and error.
classification_cost(input=output, label=label)
cls = classification_cost(input=output, label=label)
outputs(cls)
