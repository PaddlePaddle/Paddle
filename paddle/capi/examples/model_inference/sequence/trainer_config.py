from paddle.trainer_config_helpers import *

WORD_DIM = 3000

sentence = data_layer(name='sentence', size=WORD_DIM)
sentence_embedding = embedding_layer(
    input=sentence,
    size=64,
    param_attr=ParameterAttribute(
        initial_max=1.0, initial_min=0.5))
lstm = simple_lstm(input=sentence_embedding, size=64)
lstm_last = last_seq(input=lstm)
outputs(fc_layer(input=lstm_last, size=2, act=SoftmaxActivation()))
