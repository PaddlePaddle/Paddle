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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
using IntArray = paddle::experimental::IntArray;
namespace paddle {
namespace prim {
// We put some api like utils here
template <typename T>
paddle::experimental::Tensor empty(const paddle::experimental::IntArray& shape,
                                   paddle::experimental::DataType dype,
                                   const paddle::Place& place);

template <typename T>
paddle::experimental::Tensor empty_like(const paddle::experimental::Tensor& x,
                                        paddle::experimental::DataType dtype,
                                        const paddle::Place& place);
template <typename T>
void by_pass(const paddle::experimental::Tensor& x,
             paddle::experimental::Tensor* out);

// Returns reduced axes for param@shape, which broadcast with or broadcast to
// param@ref_shape.
// Note: Broadcast semantics is bidirectional. This method only returns reduced
// axes for direction shape to ref_shape.
static phi::DDim get_reduce_dims(const phi::DDim& shape,
                                 const phi::DDim& ref_shape) {
  std::vector<int64_t> result;
  auto src_shape = phi::vectorize(shape);
  auto dst_shape = phi::vectorize(ref_shape);

  // Align rank
  if (src_shape.size() > dst_shape.size()) {
    auto size = src_shape.size() - dst_shape.size();
    for (std::size_t i = 0; i < size; i++) {
      dst_shape.insert(std::begin(dst_shape), 1);
    }
  } else {
    auto size = dst_shape.size() - src_shape.size();
    for (std::size_t i = 0; i < size; i++) {
      src_shape.insert(std::begin(src_shape), 1);
    }
  }

  // Reduced axes
  for (std::size_t i = 0; i < src_shape.size(); i++) {
    if (src_shape[i] == 1 && dst_shape[i] > 1) {
      result.push_back(i);
    } else if (src_shape[i] != dst_shape[i] && src_shape[i] != 1 &&
               dst_shape[i] != 1) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The input arguments of GetReduceDims are not broadcastable. The "
          "size of parameter shape[%d]:%d can not broadcast with the size "
          "of parameter ref_shape[%d]:%d.",
          i,
          src_shape[i],
          i,
          dst_shape[i]));
    }
  }
  return phi::make_ddim(result);
}

}  // namespace prim
}  // namespace paddle
