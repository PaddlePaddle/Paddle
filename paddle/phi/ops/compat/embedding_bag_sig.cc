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

KernelSignature EmbeddingBagOpArgumentMapping(const ArgumentMappingContext& ctx) {
    return KernelSignature("embedding_bag",{"input","params","weight"},{"mode"},{"out"});

}

KernelSignature EmbeddingBagGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
    return KernelSignature("embedding_bag_grad",{"input","params","weight","out@GRAD"},{"mode"},{"params@GRAD","weight@GRAD"});
}




} // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(embedding_bag, embedding_bag);
PD_REGISTER_BASE_KERNEL_NAME(embedding_bag_grad, embedding_bag_grad);

PD_REGISTER_ARG_MAPPING_FN(embedding_bag, phi::EmbeddingBagOpArgumentMapping)
PD_REGISTER_ARG_MAPPING_FN(embedding_bag_grad,phi::EmbeddingBagGradOpArgumentMapping)