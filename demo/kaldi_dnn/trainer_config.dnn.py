# edit-mode: -*- python -*-

from paddle.trainer_config_helpers import *

trn = 'data/train.list'
tst = 'data/test.list'
process = 'process'

define_py_data_sources2(train_list=trn,
                        test_list=tst,
                        module="dataprovider_ark",
                        obj=process,
                        args={"num_senone" : 3513})

batch_size = 256
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
cls = classification_cost(input=output, label=label)
outputs(cls)
