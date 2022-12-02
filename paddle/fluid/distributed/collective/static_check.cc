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

#include "paddle/fluid/distributed/collective/static_check.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

void CommStaticCheck::SingleTensor(const phi::DenseTensor& tensor,
                                   int rank,
                                   int world_size) {
  // place check
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(tensor.place()),
      true,
      platform::errors::InvalidArgument("Tensor should be in GPU place."));
  // rank check
  PADDLE_ENFORCE_GE(rank,
                    0,
                    platform::errors::InvalidArgument(
                        "Rank should be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      rank,
      world_size,
      platform::errors::InvalidArgument("Rank is out of the process group."));
}

void CommStaticCheck::SameShape(const phi::DenseTensor& out_tensor,
                                const phi::DenseTensor& in_tensor,
                                int rank,
                                int world_size) {
  CustomShape(out_tensor,
              in_tensor,
              rank,
              world_size,
              /*out_size_factor*/ 1,
              /*in_size_factor*/ 1);
}

void CommStaticCheck::ScatterLikeShape(const phi::DenseTensor& out_tensor,
                                       const phi::DenseTensor& in_tensor,
                                       int rank,
                                       int world_size) {
  CustomShape(out_tensor,
              in_tensor,
              rank,
              world_size,
              /*out_size_factor*/ world_size,
              /*in_size_factor*/ 1);
}

void CommStaticCheck::GatherLikeShape(const phi::DenseTensor& out_tensor,
                                      const phi::DenseTensor& in_tensor,
                                      int rank,
                                      int world_size) {
  CustomShape(out_tensor,
              in_tensor,
              rank,
              world_size,
              /*out_size_factor*/ 1,
              /*in_size_factor*/ world_size);
}

void CommStaticCheck::CustomShape(const phi::DenseTensor& out_tensor,
                                  const phi::DenseTensor& in_tensor,
                                  int rank,
                                  int world_size,
                                  int out_size_factor,
                                  int in_size_factor) {
  // place check
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(out_tensor.place()),
      true,
      phi::errors::InvalidArgument("Output tensor should be in GPU place."));
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(in_tensor.place()),
      true,
      phi::errors::InvalidArgument("Input tensor should be in GPU place."));
  // rank check
  PADDLE_ENFORCE_GE(rank,
                    0,
                    phi::errors::InvalidArgument(
                        "Rank should be greater than or equal to 0."));
  PADDLE_ENFORCE_LT(
      rank,
      world_size,
      phi::errors::InvalidArgument("Rank is out of the process group."));
  // shape check
  int64_t out_size = out_tensor.numel();
  PADDLE_ENFORCE_GT(out_size,
                    0,
                    phi::errors::InvalidArgument(
                        "Size of output tensor should be greater than 0."));
  int64_t in_size = in_tensor.numel();
  PADDLE_ENFORCE_GT(in_size,
                    0,
                    phi::errors::InvalidArgument(
                        "Size of input tensor should be greater than 0."));
  PADDLE_ENFORCE_EQ(
      out_size * out_size_factor,
      in_size * in_size_factor,
      phi::errors::InvalidArgument(
          "Input and output tensors should have matching sizes."));
  // dtype check
  PADDLE_ENFORCE_EQ(
      out_tensor.dtype(),
      in_tensor.dtype(),
      phi::errors::InvalidArgument(
          "Input and output tensors should have the same data type."));
}

}  //  namespace distributed
}  //  namespace paddle
