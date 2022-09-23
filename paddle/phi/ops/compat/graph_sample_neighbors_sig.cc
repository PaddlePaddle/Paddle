/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature GraphSampleNeighborsOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("graph_sample_neighbors",
                         {"Row", "Col_Ptr", "X", "Eids", "Perm_Buffer"},
                         {"sample_size", "return_eids", "flag_perm_buffer"},
                         {"Out", "Out_Count", "Out_Eids"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(graph_sample_neighbors,
                           phi::GraphSampleNeighborsOpArgumentMapping);
