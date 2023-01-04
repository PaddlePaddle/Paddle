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

#pragma once

#include <cstdint>
#include <vector>

#include "paddle/phi/backends/gpu/forwards.h"

#ifdef PADDLE_WITH_HIP
using gpuStream_t = hipStream_t;
#else
using gpuStream_t = cudaStream_t;
#endif

// forward declarations
namespace phi {
class DenseTensor;
}

namespace paddle {
namespace distributed {

struct CommStaticCheck {
  static void CheckRank(int rank, int world_size);

  static void CheckPlace(const phi::DenseTensor& tensor);

  static void CheckPlace(const phi::DenseTensor& out_tensor,
                         const phi::DenseTensor& in_tensor);

  static void CheckDataType(const phi::DenseTensor& out_tensor,
                            const phi::DenseTensor& in_tensor);

  static void CheckShape(const phi::DenseTensor& tensor);

  static void CheckShape(const phi::DenseTensor& out_tensor,
                         const phi::DenseTensor& in_tensor,
                         int out_size_factor,
                         int in_size_factor);

  static void CheckShape(const phi::DenseTensor& out_tensor,
                         const phi::DenseTensor& in_tensor,
                         int dst_rank,
                         int cur_rank,
                         int world_size,
                         int out_size_factor,
                         int in_size_factor);

  // for p2p
  static void CheckShape(const phi::DenseTensor& tensor,
                         int rank,
                         int world_size);

  // for collective
  static void SameShape(const phi::DenseTensor& out_tensor,
                        const phi::DenseTensor& in_tensor,
                        int dst_rank,
                        int cur_rank,
                        int world_size);

  static void ScatterLikeShape(const phi::DenseTensor& out_tensor,
                               const phi::DenseTensor& in_tensor,
                               int dst_rank,
                               int cur_rank,
                               int world_size);

  static void GatherLikeShape(const phi::DenseTensor& out_tensor,
                              const phi::DenseTensor& in_tensor,
                              int dst_rank,
                              int cur_rank,
                              int world_size);
};

struct CommDynamicCheck {
  static void CheckDataType(const phi::DenseTensor& tensor, int64_t dtype);

  static void CheckDataType(const phi::DenseTensor& tensor,
                            int root_rank,
                            int cur_rank,
                            ncclComm_t comm);

  static void CheckShape(const phi::DenseTensor& tensor, int64_t shape);

  static void CheckShape(const phi::DenseTensor& tensor,
                         int root_rank,
                         int cur_rank,
                         ncclComm_t comm);

  static void CheckShape(const phi::DenseTensor& out_tensor,
                         const phi::DenseTensor& in_tensor,
                         const std::vector<int64_t>& in_size_each_rank,
                         int cur_rank,
                         int world_size,
                         ncclComm_t comm);

 private:
  // `0` represents default stream for both cuda & hip
  static constexpr gpuStream_t kDefaultStream = 0;
};

}  // namespace distributed
}  // namespace paddle
