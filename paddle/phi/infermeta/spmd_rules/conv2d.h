/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo Conv2dInferSpmdBase(const DistMetaTensor& input,
                             const DistMetaTensor& filter);

SpmdInfo Conv2dInferSpmdReverseBase(const DistMetaTensor& input,
                                    const DistMetaTensor& filter,
                                    const DistMetaTensor& output);

SpmdInfo Conv2dGradInferSpmdBase(const DistMetaTensor& input,
                                 const DistMetaTensor& filter,
                                 const DistMetaTensor& output_grad);

SpmdInfo Conv2dInferSpmd(const DistMetaTensor& input,
                         const DistMetaTensor& filter,
                         const std::vector<int>& strides = {1, 1},
                         const std::vector<int>& paddings = {0, 0},
                         const std::string& padding_algorithm = "EXPLICIT",
                         const std::vector<int>& dilations = {1, 1},
                         int groups = 1,
                         const std::string& data_format = "NCHW");

SpmdInfo Conv2dInferSpmdReverse(
    const DistMetaTensor& input,
    const DistMetaTensor& filter,
    const DistMetaTensor& output,
    const std::vector<int>& strides = {1, 1},
    const std::vector<int>& paddings = {0, 0},
    const std::string& padding_algorithm = "EXPLICIT",
    const std::vector<int>& dilations = {1, 1},
    int groups = 1,
    const std::string& data_format = "NCHW");

SpmdInfo Conv2dGradInferSpmd(const DistMetaTensor& input,
                             const DistMetaTensor& filter,
                             const DistMetaTensor& output_grad,
                             const std::vector<int>& strides = {1, 1},
                             const std::vector<int>& paddings = {0, 0},
                             const std::string& padding_algorithm = "EXPLICIT",
                             const std::vector<int>& dilations = {1, 1},
                             int groups = 1,
                             const std::string& data_format = "NCHW");

}  // namespace distributed
}  // namespace phi
