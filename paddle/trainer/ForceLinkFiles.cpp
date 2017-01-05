/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/ForceLink.h"

/**
 * this file used for static linking libraries, to enable all InitFunction in
 * Paddle GServer.
 */
PADDLE_ENABLE_FORCE_LINK_FILE(base_data_providers);
PADDLE_ENABLE_FORCE_LINK_FILE(multi_data_dp);
PADDLE_ENABLE_FORCE_LINK_FILE(proto_dp);
#ifndef PADDLE_NO_PYTHON
PADDLE_ENABLE_FORCE_LINK_FILE(py_dp);
PADDLE_ENABLE_FORCE_LINK_FILE(py_dp2);
#endif
PADDLE_ENABLE_FORCE_LINK_FILE(activations);
PADDLE_ENABLE_FORCE_LINK_FILE(base_evaluators);
PADDLE_ENABLE_FORCE_LINK_FILE(ctc_evaluator);
PADDLE_ENABLE_FORCE_LINK_FILE(chunck_evaluator);

PADDLE_ENABLE_FORCE_LINK_FILE(addto_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(agent_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(average_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(batch_norm_base_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(batch_normalization_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(bilinear_interp_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(block_expand_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(concatenate_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(context_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(conv_base_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(conv_operator);
PADDLE_ENABLE_FORCE_LINK_FILE(conv_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(conv_shift_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(convex_combination_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(cos_sim_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(cos_sim_vec_mat_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(cost_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(data_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(data_norm_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(dot_mul_operator);
PADDLE_ENABLE_FORCE_LINK_FILE(dot_mul_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(eos_id_check_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(expand_conv_base_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(expand_conv_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(expand_conv_trans_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(expand_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(feature_map_expand_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(full_matrix_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(fully_connected_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(gated_recurrent_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(get_output_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(gru_compute);
PADDLE_ENABLE_FORCE_LINK_FILE(gru_step_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(hierarchical_sigmoid_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(identity_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(interpolation_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(layer);
PADDLE_ENABLE_FORCE_LINK_FILE(lstm_compute);
PADDLE_ENABLE_FORCE_LINK_FILE(lstm_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(lstm_step_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(max_id_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(max_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(max_out_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(mixed_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(multinomial_sampler);
PADDLE_ENABLE_FORCE_LINK_FILE(multiplex_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(norm_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(norm_projection_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(operator);
PADDLE_ENABLE_FORCE_LINK_FILE(outer_prod_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(parameter_relu_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(pool_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(pool_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(pool_projection_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(power_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(print_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(prior_box);
PADDLE_ENABLE_FORCE_LINK_FILE(projection);
PADDLE_ENABLE_FORCE_LINK_FILE(recurrent_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(recurrent_layer_group);
PADDLE_ENABLE_FORCE_LINK_FILE(resize_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sampling_id_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(scaling_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(scaling_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(selective_fully_connected_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sequence_concat_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sequence_last_instance_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sequence_pool_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sequence_reshape_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sequence_to_batch);
PADDLE_ENABLE_FORCE_LINK_FILE(slope_intercept_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(spatial_pyramid_pool_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sub_sequence_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(sum_to_one_norm_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(table_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(tensor_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(trans_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(transposed_full_matrix_projection);
PADDLE_ENABLE_FORCE_LINK_FILE(validation_layer);

PADDLE_ENABLE_FORCE_LINK_FILE(crf_decoding);
PADDLE_ENABLE_FORCE_LINK_FILE(crf);
PADDLE_ENABLE_FORCE_LINK_FILE(ctc);
PADDLE_ENABLE_FORCE_LINK_FILE(mdlstm);
PADDLE_ENABLE_FORCE_LINK_FILE(nce);
PADDLE_ENABLE_FORCE_LINK_FILE(warp_ctc);

#ifndef PADDLE_ONLY_CPU
PADDLE_ENABLE_FORCE_LINK_FILE(cudnn_batch_norm_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(cudnn_conv_layer);
PADDLE_ENABLE_FORCE_LINK_FILE(cudnn_pool_layer);
#endif

PADDLE_ENABLE_FORCE_LINK_FILE(cross_map_norm_ops);
PADDLE_ENABLE_FORCE_LINK_FILE(context_proj_ops);
