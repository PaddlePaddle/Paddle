# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PROTOBUF3_NEEDED_TESTING_LIST = [
    'test_dist_fleet_ps11',
    'test_dist_fleet_ps12',
    'test_dataloader_dataset',
    'test_ema_fleet',
    'test_fleet_base2',
    'test_fleet_base3',
    'test_communicator_geo_deprecated',
    'test_communicator_async',
    'test_dist_fleet_a_sync_optimizer_async',
    'test_dist_fleet_a_sync_optimizer_auto',
    'test_dist_fleet_a_sync_optimizer_sync',
    'test_dist_fleet_a_sync_optimizer_geo_deprecated',
    'test_dist_fleet_a_sync_optimizer_auto_geo',
    'test_dist_fleet_a_sync_optimizer_auto_async',
    'test_fleet_fp16_allreduce_meta_optimizer',
    'test_dist_fleet_ctr',
    'test_dist_fleet_ctr2',
    'test_dist_fleet_decay',
    'test_dist_fleet_geo',
    'test_dist_fleet_heter_program',
    'test_dist_fleet_ps',
    'test_dist_fleet_ps10',
    'test_dist_fleet_ps13',
    'test_dist_fleet_ps2',
    'test_dist_fleet_ps3',
    'test_dist_fleet_ps4',
    'test_dist_fleet_ps5',
    'test_dist_fleet_ps6',
    'test_dist_fleet_ps7',
    'test_dist_fleet_ps8',
    'test_dist_fleet_ps9',
    'test_dist_fleet_simnet',
    'test_dist_fleet_sparse_embedding_ctr',
    'test_dist_sparse_tensor_load_adagrad',
    'test_dist_sparse_tensor_load_adam',
    'test_dist_sparse_tensor_load_ftrl',
    'test_dist_sparse_tensor_load_momentum',
    'test_dist_sparse_tensor_load_rmsprop',
    'test_dist_sparse_tensor_load_sgd_deprecated',
    'test_communicator_sync',
    'test_dist_fuse_adam_pass',
    'test_dist_fuse_bn_act_pass',
    'test_dist_fuse_bn_add_act_pass',
    'test_dist_fuse_momentum_pass',
    'test_dist_fuse_relu_depthwise_conv_pass',
    'test_dist_fuse_sgd_pass',
    'test_dist_inplace_addto_pass',
    'test_fleet_sharding_meta_optimizer',
    'test_rnn_dp',
    'test_communicator_half_async',
    'test_fleet_graph_executor',
    'test_fleet_localsgd_meta_optimizer',
    'test_fleet_lars_meta_optimizer',
    'test_fleet_pipeline_meta_optimizer',
    'test_fleet_gradient_merge_meta_optimizer',
    'test_fleet_amp_init',
    'test_fleet_raw_program_meta_optimizer',
    'test_fleet_dgc_meta_optimizer',
    'test_fleet_lamb_meta_optimizer',
    'test_fleet_pipeline_meta_optimizer_with_recompute',
    'test_fleet_hybrid_meta_optimizer',
    'test_fleet_amp_meta_optimizer',
    'test_fleet_recompute_meta_optimizer',
    'test_fleet_with_asp_static',
    'test_fleet_with_asp_sharding',
]

if __name__ == "__main__":
    for test in PROTOBUF3_NEEDED_TESTING_LIST:
        print(test)
