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

#include "glog/logging.h"

#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using Tensor = DenseTensor;

template <typename DeviceContext, typename T>
inline void ResizeToChannelFirst(const DeviceContext& context,
                                 const Tensor* input,
                                 Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = vectorize(input->dims());
    in_dims_vec[1] = input->dims()[4];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    in_dims_vec[4] = input->dims()[3];
    transformed_input->Resize(make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = vectorize(input->dims());
    in_dims_vec[1] = input->dims()[3];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    transformed_input->Resize(make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  } else if (dim == 1) {
    transformed_input->Resize(input->dims());

    auto in_dims_vec = vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[1];
    transformed_input->Resize(make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  }
}

template <typename DeviceContext, typename T>
inline void ResizeToChannelLast(const DeviceContext& context,
                                const Tensor* input,
                                Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[3];
    in_dims_vec[3] = input->dims()[4];
    in_dims_vec[4] = input->dims()[1];
    transformed_input->Resize(make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);

  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[3];
    in_dims_vec[3] = input->dims()[1];
    transformed_input->Resize(make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  } else if (dim == 1) {
    transformed_input->Resize(input->dims());

    auto in_dims_vec = vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[1];
    transformed_input->Resize(make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelFirst(const DeviceContext& context,
                                const Tensor* input,
                                Tensor* transformed_input) {
  VLOG(5) << "Why am I called?";
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    std::vector<int> axis{0, 4, 1, 2, 3};
    phi::funcs::Transpose<DeviceContext, T, 5> trans5;
    trans5(context, *input, transformed_input, axis);

  } else if (dim == 2) {
    std::vector<int> axis{0, 3, 1, 2};
    phi::funcs::Transpose<DeviceContext, T, 4> trans4;
    trans4(context, *input, transformed_input, axis);
  } else if (dim == 1) {
    std::vector<int> axis{0, 2, 1};
    phi::funcs::Transpose<DeviceContext, T, 3> trans3;
    trans3(context, *input, transformed_input, axis);
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelLast(const DeviceContext& context,
                               const Tensor* input,
                               Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    std::vector<int> axis{0, 2, 3, 4, 1};
    phi::funcs::Transpose<DeviceContext, T, 5> trans5;
    trans5(context, *input, transformed_input, axis);

  } else if (dim == 2) {
    std::vector<int> axis{0, 2, 3, 1};
    phi::funcs::Transpose<DeviceContext, T, 4> trans4;
    trans4(context, *input, transformed_input, axis);
  } else if (dim == 1) {
    std::vector<int> axis{0, 2, 1};
    phi::funcs::Transpose<DeviceContext, T, 3> trans3;
    trans3(context, *input, transformed_input, axis);
  }
}

}  // namespace phi
