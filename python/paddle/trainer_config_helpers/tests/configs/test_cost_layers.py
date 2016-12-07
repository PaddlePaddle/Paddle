from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

seq_in = data_layer(name='input', size=200)
labels = data_layer(name='labels', size=5000)

probs = data_layer(name='probs', size=10)
xe_label = data_layer(name='xe-label', size=10)

hidden = fc_layer(input=seq_in, size=4)
outputs(
    ctc_layer(
        input=seq_in, label=labels),
    warp_ctc_layer(
        input=seq_in, label=labels, blank=0),
    crf_layer(
        input=hidden, label=data_layer(
            name='crf_label', size=4)),
    rank_cost(
        left=data_layer(
            name='left', size=1),
        right=data_layer(
            name='right', size=1),
        label=data_layer(
            name='label', size=1)),
    lambda_cost(
        input=data_layer(
            name='list_feature', size=100),
        score=data_layer(
            name='list_scores', size=1)),
    cross_entropy(
        input=probs, label=xe_label),
    cross_entropy_with_selfnorm(
        input=probs, label=xe_label),
    huber_cost(
        input=data_layer(
            name='huber_probs', size=1),
        label=data_layer(
            name='huber_label', size=1)),
    multi_binary_label_cross_entropy(
        input=probs, label=xe_label),
    sum_cost(input=hidden))
