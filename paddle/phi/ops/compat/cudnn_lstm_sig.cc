// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature CudnnLSTMOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "cudnn_lstm",
      {"Input", "InitH", "InitC", "W", "WeightList", "SequenceLength"},
      {"dropout_prob",
       "is_bidirec",
       "hidden_size",
       "num_layers",
       "is_test",
       "seed"},
      {"Out", "LastH", "LastC", "Reserve", "StateOut"});
}

KernelSignature CudnnLSTMGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "cudnn_lstm_grad",
      {"Input",
       "InitH",
       "InitC",
       "WeightList",
       "SequenceLength",
       "Out",
       "Reserve",
       "StateOut",
       "Out@GRAD",
       "LastH@GRAD",
       "LastC@GRAD"},
      {"dropout_prob",
       "is_bidirec",
       "hidden_size",
       "num_layers",
       "is_test",
       "seed"},
      {"Input@GRAD", "InitH@GRAD", "InitC@GRAD", "WeightList@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(cudnn_lstm, phi::CudnnLSTMOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(cudnn_lstm_grad,
                           phi::CudnnLSTMGradOpArgumentMapping);
