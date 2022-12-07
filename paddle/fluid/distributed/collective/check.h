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

// forward declaration to reduce deps
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

}  // namespace distributed
}  // namespace paddle
