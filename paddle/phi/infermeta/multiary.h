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

#pragma once

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"
namespace phi {

std::vector<DDim> GetMetaTensorsDim(const std::vector<MetaTensor*>& tensors);

void BilinearTensorProductInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& weight,
                                    paddle::optional<const MetaTensor&> bias,
                                    MetaTensor* out,
                                    MetaConfig config = MetaConfig());

void BroadcastTensorsInferMeta(const std::vector<MetaTensor*>& x,
                               std::vector<MetaTensor*> out);

void ConcatInferMeta(const std::vector<MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out);
}  // namespace phi
