// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdlib>
#include <iostream>
#include <vector>

#include "paddle/extension.h"

template <typename data_t>
void assign_cpu_kernel(const data_t* x_data,
                       data_t* out_data,
                       int64_t x_numel) {
  for (int i = 0; i < x_numel; ++i) {
    out_data[i] = x_data[i];
  }
}

void CheckAllForwardAttrs(const bool& bool_attr,
                          const int& int_attr,
                          const float& float_attr,
                          const int64_t& int64_attr,
                          const std::string& str_attr,
                          const std::vector<int>& int_vec_attr,
                          const std::vector<float>& float_vec_attr,
                          const std::vector<int64_t>& int64_vec_attr,
                          const std::vector<std::string>& str_vec_attr) {
  if (bool_attr != true) {
    throw std::runtime_error("bool_attr value error.");
  }
  if (int_attr != 10) {
    throw std::runtime_error("int_attr value error.");
  }
  if (std::abs(float_attr - 3.14) > 1e-6) {
    throw std::runtime_error("float_attr value error.");
  }
  if (int64_attr != 10000000000) {
    throw std::runtime_error("int64_attr value error.");
  }
  if (str_attr != "StrAttr") {
    throw std::runtime_error("str_attr value error.");
  }

  if (int_vec_attr.size() != 3) {
    throw std::runtime_error("int_vec_attr size error.");
  } else {
    for (auto& value : int_vec_attr) {
      if (value != 10) {
        throw std::runtime_error("int_vec_attr value error.");
      }
    }
  }

  if (float_vec_attr.size() != 3) {
    throw std::runtime_error("float_vec_attr size error.");
  } else {
    for (auto& value : float_vec_attr) {
      if (std::abs(value - 3.14) > 1e-6) {
        throw std::runtime_error("float_vec_attr value error.");
      }
    }
  }

  if (int64_vec_attr.size() != 3) {
    throw std::runtime_error("int64_vec_attr size error.");
  } else {
    for (auto& value : int64_vec_attr) {
      if (value != 10000000000) {
        throw std::runtime_error("int64_vec_attr value error.");
      }
    }
  }

  if (str_vec_attr.size() != 3) {
    throw std::runtime_error("str_vec_attr size error.");
  } else {
    for (auto& value : str_vec_attr) {
      if (value != "StrAttr") {
        throw std::runtime_error("str_vec_attr value error.");
      }
    }
  }
}

void CheckAllBackwardAttrs(const int& int_attr,
                           const std::vector<float>& float_vec_attr,
                           const std::vector<std::string>& str_vec_attr) {
  if (int_attr != 10) {
    throw std::runtime_error("int_attr value error.");
  }

  if (float_vec_attr.size() != 3) {
    throw std::runtime_error("float_vec_attr size error.");
  } else {
    for (auto& value : float_vec_attr) {
      if (std::abs(value - 3.14) > 1e-6) {
        throw std::runtime_error("float_vec_attr value error.");
      }
    }
  }

  if (str_vec_attr.size() != 3) {
    throw std::runtime_error("str_vec_attr size error.");
  } else {
    for (auto& value : str_vec_attr) {
      if (value != "StrAttr") {
        throw std::runtime_error("str_vec_attr value error.");
      }
    }
  }
}

std::vector<paddle::Tensor> AttrTestForward(
    const paddle::Tensor& x,
    bool bool_attr,
    int int_attr,
    float float_attr,
    int64_t int64_attr,
    std::string str_attr,
    std::vector<int> int_vec_attr,
    std::vector<float> float_vec_attr,
    std::vector<int64_t> int64_vec_attr,
    std::vector<std::string> str_vec_attr) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  // Check attrs value
  CheckAllForwardAttrs(bool_attr,
                       int_attr,
                       float_attr,
                       int64_attr,
                       str_attr,
                       int_vec_attr,
                       float_vec_attr,
                       int64_vec_attr,
                       str_vec_attr);

  return {out};
}

// The attrs of backward op must be the subset of attrs of forward op
std::vector<paddle::Tensor> AttrTestBackward(
    const paddle::Tensor& grad_out,
    int int_attr,
    std::vector<float> float_vec_attr,
    std::vector<std::string> str_vec_attr) {
  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, grad_out.shape());

  PD_DISPATCH_FLOATING_TYPES(grad_out.type(), "assign_cpu_kernel", ([&] {
                               assign_cpu_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(),
                                   grad_out.size());
                             }));

  CheckAllBackwardAttrs(int_attr, float_vec_attr, str_vec_attr);

  return {grad_x};
}

std::vector<paddle::Tensor> ConstAttrTestForward(
    const paddle::Tensor& x,
    const bool& bool_attr,
    const int& int_attr,
    const float& float_attr,
    const int64_t& int64_attr,
    const std::string& str_attr,
    const std::vector<int>& int_vec_attr,
    const std::vector<float>& float_vec_attr,
    const std::vector<int64_t>& int64_vec_attr,
    const std::vector<std::string>& str_vec_attr) {
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "assign_cpu_kernel", ([&] {
        assign_cpu_kernel<data_t>(
            x.data<data_t>(), out.mutable_data<data_t>(), x.size());
      }));

  // Check attrs value
  CheckAllForwardAttrs(bool_attr,
                       int_attr,
                       float_attr,
                       int64_attr,
                       str_attr,
                       int_vec_attr,
                       float_vec_attr,
                       int64_vec_attr,
                       str_vec_attr);

  return {out};
}

// The attrs of backward op must be the subset of attrs of forward op
std::vector<paddle::Tensor> ConstAttrTestBackward(
    const paddle::Tensor& grad_out,
    const int& int_attr,
    const std::vector<float>& float_vec_attr,
    const std::vector<std::string>& str_vec_attr) {
  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, grad_out.shape());

  PD_DISPATCH_FLOATING_TYPES(grad_out.type(), "assign_cpu_kernel", ([&] {
                               assign_cpu_kernel<data_t>(
                                   grad_out.data<data_t>(),
                                   grad_x.mutable_data<data_t>(),
                                   grad_out.size());
                             }));

  CheckAllBackwardAttrs(int_attr, float_vec_attr, str_vec_attr);

  return {grad_x};
}

PD_BUILD_OP(attr_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"bool_attr: bool",
            "int_attr: int",
            "float_attr: float",
            "int64_attr: int64_t",
            "str_attr: std::string",
            "int_vec_attr: std::vector<int>",
            "float_vec_attr: std::vector<float>",
            "int64_vec_attr: std::vector<int64_t>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestForward));

PD_BUILD_GRAD_OP(attr_test)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"int_attr: int",
            "float_vec_attr: std::vector<float>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestBackward));

PD_BUILD_OP(const_attr_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"bool_attr: bool",
            "int_attr: int",
            "float_attr: float",
            "int64_attr: int64_t",
            "str_attr: std::string",
            "int_vec_attr: std::vector<int>",
            "float_vec_attr: std::vector<float>",
            "int64_vec_attr: std::vector<int64_t>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestForward));

PD_BUILD_GRAD_OP(const_attr_test)
    .Inputs({paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({"int_attr: int",
            "float_vec_attr: std::vector<float>",
            "str_vec_attr: std::vector<std::string>"})
    .SetKernelFn(PD_KERNEL(AttrTestBackward));
