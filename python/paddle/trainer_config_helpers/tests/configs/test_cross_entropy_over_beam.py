#!/usr/bin/env python
#coding=utf-8

from paddle.trainer_config_helpers import *
beam_size = 5

# the first beam expansion.
sentence_states = data_layer(name="sentence_states", size=32)
sentence_scores = data_layer(name="sentence_scores", size=1)
topk_sentence_ids = kmax_seq_score_layer(
    input=sentence_scores, beam_size=beam_size)

# the second beam expansion.
topk_sen = sub_nested_seq_layer(
    input=sentence_states, selected_indices=topk_sentence_ids)
start_pos_scores = fc_layer(input=topk_sen, size=1, act=LinearActivation())
topk_start_pos_ids = kmax_seq_score_layer(
    input=sentence_scores, beam_size=beam_size)

# the final beam expansion.
topk_start_spans = seq_slice_layer(
    input=topk_sen, starts=topk_start_pos_ids, ends=None)
end_pos_scores = fc_layer(
    input=topk_start_spans, size=1, act=LinearActivation())
topk_end_pos_ids = kmax_seq_score_layer(
    input=end_pos_scores, beam_size=beam_size)

# define the cost
sentence_idx = data_layer(name="sentences_ids", size=1)
start_idx = data_layer(name="start_ids", size=1)
end_idx = data_layer(name="end_ids", size=1)
cost = cross_entropy_over_beam(input=[
    BeamInput(
        candidate_scores=sentence_scores,
        selected_candidates=topk_sentence_ids,
        gold=sentence_idx), BeamInput(
            candidate_scores=start_pos_scores,
            selected_candidates=topk_start_pos_ids,
            gold=start_idx), BeamInput(
                candidate_scores=end_pos_scores,
                selected_candidates=topk_end_pos_ids,
                gold=end_idx)
])

outputs(cost)
