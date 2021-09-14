// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/top/api/include/tensor.h"
#include "paddle/fluid/imperative/layer.h"
std::vector<std::shared_ptr<paddle::imperative::VarBase>> TensorsToVarBases(const pt::Tensor& tensor);
std::vector<std::shared_ptr<paddle::imperative::VarBase>> TensorsToVarBases(const std::vector<pt::Tensor>& tensors);
std::vector<pt::Tensor> VarBasesToTensors(const std::shared_ptr<paddle::imperative::VarBase>& var_base);
std::vector<pt::Tensor> VarBasesToTensors(const std::vector<std::shared_ptr<paddle::imperative::VarBase>>& var_bases);

