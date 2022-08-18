// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

KernelSignature RnnOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("rnn",
                         {"Input", "PreState", "WeightList", "SequenceLength"},
                         {"dropout_prob",
                          "is_bidirec",
                          "input_size",
                          "hidden_size",
                          "num_layers",
                          "mode",
                          "seed",
                          "is_test"},
                         {"Out", "DropoutState", "State", "Reserve"});
}

KernelSignature RnnGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("rnn_grad",
                         {"Input",
                          "PreState",
                          "WeightList",
                          "SequenceLength",
                          "Out",
                          "DropoutState",
                          "Reserve",
                          "Out@GRAD",
                          "State@GRAD"},
                         {"dropout_prob",
                          "is_bidirec",
                          "input_size",
                          "hidden_size",
                          "num_layers",
                          "mode",
                          "seed",
                          "is_test"},
                         {"Input@GRAD", "PreState@GRAD", "WeightList@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(rnn, phi::RnnOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(rnn_grad, phi::RnnGradOpArgumentMapping);
