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
// These method don't need to be specified
static phi::DDim get_reduce_dims(const phi::DDim& dout_dims,
                                 const phi::DDim& in_dims) {
  std::vector<int64_t> result;
  int bat = dout_dims.size() - in_dims.size();
  for (int i = 0; i < bat; ++i) {
    result.push_back(i);
  }
  for (int i = 0; i < in_dims.size(); ++i) {
    if (in_dims[i] == 1) {
      result.push_back(i + bat);
    } else {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          dout_dims[i + bat],
          platform::errors::InvalidArgument(
              "ReduceDims dimension mismatch. Operands could "
              "not be broadcast together with the shape of dout = [%s] and "
              "the shape of in_dims = [%s]. Received [%d] in X is not equal to "
              "[%d] in Y at i:%d.",
              dout_dims,
              in_dims,
              dout_dims[i + bat],
              in_dims[i],
              i));
    }
  }
  auto res_dims = phi::make_ddim(result);
  VLOG(4) << "Reduce Dims is: " << res_dims;
  return res_dims;
}
}  // namespace prim
}  // namespace paddle
