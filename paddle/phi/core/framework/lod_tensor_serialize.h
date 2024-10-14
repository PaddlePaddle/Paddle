// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/framework/dense_tensor_tostream.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/utils/test_macros.h"

namespace phi {

/*
 * Serialize/Deserialize phi::DenseTensor to std::ostream
 * You can pass ofstream or ostringstream to serialize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os,
                       const phi::DenseTensor& tensor,
                       const phi::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is,
                           phi::DenseTensor* tensor,
                           const phi::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is,
                           phi::DenseTensor* tensor,
                           const phi::DeviceContext& dev_ctx,
                           const size_t& seek,
                           const std::vector<int64_t>& shape);

void SerializeToStream(std::ostream& os, const phi::DenseTensor& tensor);

void DeserializeFromStream(std::istream& os, phi::DenseTensor* tensor);

}  // namespace phi
