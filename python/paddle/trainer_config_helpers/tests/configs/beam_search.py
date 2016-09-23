from paddle.trainer_config_helpers import *

settings(
    learning_rate=1e-4,
    batch_size=1000
)

dat = data_layer(name='data_in', size=100)


def rnn_step(input, embedding):
    last_time_step_output = memory(name='rnn', size=128)
    with mixed_layer(size=128, name='rnn') as simple_rnn:
        simple_rnn += full_matrix_projection(embedding)
        simple_rnn += full_matrix_projection(input)
        simple_rnn += full_matrix_projection(last_time_step_output)
    return simple_rnn


beam_gen = beam_search(name="decoder",
                       step=rnn_step,
                       input=[StaticInput(input=dat), GeneratedInput(
                           embedding_name='emb', embedding_size=128, size=128
                       )],
                       bos_id=0,
                       eos_id=1,
                       beam_size=5,
                       result_file="./generated_sequences.txt")

outputs(beam_gen)