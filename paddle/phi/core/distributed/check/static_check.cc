// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/check/static_check.h"

#include <cstdlib>
#include <cstring>

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi::distributed {

void CommStaticCheck::CheckRank(int rank, int world_size) {
  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::InvalidArgument(
                        "Rank should be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      rank,
      world_size,
      common::errors::InvalidArgument("Rank is out of the process group."));
}

void CommStaticCheck::CheckPlace(const phi::DenseTensor& tensor,
                                 phi::AllocationType place) {
  PADDLE_ENFORCE_EQ(
      tensor.place().GetType(),
      place,
      common::errors::InvalidArgument("Tensor should be in backend's place."));
}

void CommStaticCheck::CheckPlace(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 phi::AllocationType place) {
  CheckPlace(out_tensor, place);
  CheckPlace(in_tensor, place);
  PADDLE_ENFORCE_EQ(
      out_tensor.place(),
      in_tensor.place(),
      common::errors::InvalidArgument(
          "Input and output tensors should be on the same place."));
}

void CommStaticCheck::CheckDataType(const phi::DenseTensor& out_tensor,
                                    const phi::DenseTensor& in_tensor) {
  PADDLE_ENFORCE_EQ(
      out_tensor.dtype(),
      in_tensor.dtype(),
      common::errors::InvalidArgument(
          "Input and output tensors should have the same data type."));
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& tensor) {
  PADDLE_ENFORCE_GT(tensor.numel(),
                    0,
                    common::errors::InvalidArgument(
                        "Size of tensor should be greater than 0."));
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 int out_size_factor,
                                 int in_size_factor) {
  CheckShape(out_tensor);
  CheckShape(in_tensor);
  int64_t out_size = out_tensor.numel(), in_size = in_tensor.numel();
  PADDLE_ENFORCE_EQ(
      out_size * out_size_factor,
      in_size * in_size_factor,
      common::errors::InvalidArgument(
          "Input and output tensors should have matching sizes. "
          "out_size=%ld, out_size_factor=%d, in_size=%ld, in_size_factor=%d",
          out_size,
          out_size_factor,
          in_size,
          in_size_factor));
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& out_tensor,
                                 const phi::DenseTensor& in_tensor,
                                 int dst_rank,
                                 int cur_rank,
                                 int world_size,
                                 int out_size_factor,
                                 int in_size_factor,
                                 phi::AllocationType place) {
  CheckRank(dst_rank, world_size);
  CheckRank(cur_rank, world_size);

  CheckPlace(out_tensor, in_tensor, place);
  CheckDataType(out_tensor, in_tensor);

  if (dst_rank == cur_rank) {
    CheckShape(out_tensor, in_tensor, out_size_factor, in_size_factor);
  } else {
    CheckShape(out_tensor);
    CheckShape(in_tensor);
  }
}

void CommStaticCheck::CheckShape(const phi::DenseTensor& tensor,
                                 int rank,
                                 int world_size,
                                 phi::AllocationType place) {
  CheckPlace(tensor, place);
  CheckRank(rank, world_size);
}

void CommStaticCheck::SameShape(const phi::DenseTensor& out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int dst_rank,
                                int cur_rank,
                                int world_size,
                                phi::AllocationType place) {
  CheckShape(out_tensor,
             in_tensor,
             dst_rank,
             cur_rank,
             world_size,
             /*out_size_factor*/ 1,
             /*in_size_factor*/ 1,
             place);
}

void CommStaticCheck::ScatterLikeShape(const phi::DenseTensor& out_tensor,
                                       const phi::DenseTensor& in_tensor,
                                       int dst_rank,
                                       int cur_rank,
                                       int world_size,
                                       phi::AllocationType place) {
  CheckShape(out_tensor,
             in_tensor,
             dst_rank,
             cur_rank,
             world_size,
             /*out_size_factor*/ world_size,
             /*in_size_factor*/ 1,
             place);
}

void CommStaticCheck::GatherLikeShape(const phi::DenseTensor& out_tensor,
                                      const phi::DenseTensor& in_tensor,
                                      int dst_rank,
                                      int cur_rank,
                                      int world_size,
                                      phi::AllocationType place) {
  CheckShape(out_tensor,
             in_tensor,
             dst_rank,
             cur_rank,
             world_size,
             /*out_size_factor*/ 1,
             /*in_size_factor*/ world_size,
             place);
}

}  // namespace phi::distributed
