from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-4)

din = data_layer(name='data', size=200)

hidden = fc_layer(input=din, size=200, act=SigmoidActivation())

rnn = recurrent_layer(input=hidden, act=SigmoidActivation())

rnn2 = recurrent_layer(input=hidden, act=SigmoidActivation(), reverse=True)

lstm1_param = fc_layer(
    input=hidden, size=200 * 4, act=LinearActivation(), bias_attr=False)

lstm1 = lstmemory(input=lstm1_param, act=SigmoidActivation())

lstm2_param = fc_layer(
    input=hidden, size=200 * 4, act=LinearActivation(), bias_attr=False)

lstm2 = lstmemory(input=lstm2_param, act=SigmoidActivation(), reverse=True)

gru1_param = fc_layer(
    input=hidden, size=200 * 3, act=LinearActivation(), bias_attr=False)
gru1 = grumemory(input=gru1_param, act=SigmoidActivation())

gru2_param = fc_layer(
    input=hidden, size=200 * 3, act=LinearActivation(), bias_attr=False)
gru2 = grumemory(input=gru2_param, act=SigmoidActivation(), reverse=True)

outputs(
    last_seq(input=rnn),
    first_seq(input=rnn2),
    last_seq(input=lstm1),
    first_seq(input=lstm2),
    last_seq(input=gru1),
    first_seq(gru2))
