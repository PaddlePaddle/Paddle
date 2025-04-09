// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <vector>

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"

namespace phi {
// forward declaration
class DenseTensor;

namespace distributed {
struct BKCLDynamicCheck {
  static void CheckDataType(const phi::DenseTensor& tensor, int64_t dtype);

  static void CheckDataType(const phi::DenseTensor& tensor,
                            int root_rank,
                            int cur_rank,
                            BKCLContext_t comm);

  static void CheckShape(const phi::DenseTensor& tensor, int64_t shape);
  static void CheckShape(const phi::DenseTensor& out_tensor,
                         const phi::DenseTensor& in_tensor,
                         const std::vector<int64_t>& in_size_each_rank,
                         int cur_rank,
                         int world_size,
                         BKCLContext_t comm);
  static void CheckAlltoAllShape(
      const std::vector<phi::DenseTensor>& out_tensor,
      const std::vector<phi::DenseTensor>& in_tensor,
      int cur_rank,
      int world_size,
      BKCLContext_t comm);
};

}  //  namespace distributed
}  //  namespace phi
