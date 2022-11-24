//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace paddle {
namespace framework {
class OpKernelType;
}  // namespace framework
}  // namespace paddle

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {

struct CastDataLayout {
  CastDataLayout(const platform::DeviceContext* ctx,
                 const std::vector<int>& axis,
                 const phi::DenseTensor& in,
                 phi::DenseTensor* out)
      : in_(in), out_(out), ctx_(ctx), axis_(axis) {}

  const phi::DenseTensor in_;
  phi::DenseTensor* out_;
  const platform::DeviceContext* ctx_;
  const std::vector<int> axis_;

  template <typename T>
  void apply();
};

#ifdef PADDLE_WITH_MKLDNN
using OneDNNDataType = dnnl::memory::data_type;

void innerTransDataLayoutFromMKLDNN(DataLayout in_layout,
                                    DataLayout out_layout,
                                    const phi::DenseTensor& in,
                                    phi::DenseTensor* out,
                                    platform::Place place,
                                    bool always_copy = false);

void TransDataLayoutFromMKLDNN(const OpKernelType& kernel_type_for_var,
                               const OpKernelType& expected_kernel_type,
                               const phi::DenseTensor& in,
                               phi::DenseTensor* out);

void* GetDataFromTensor(const phi::DenseTensor& tensor, OneDNNDataType type);

#endif

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to);

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type,
                     const phi::DenseTensor& in,
                     phi::DenseTensor* out);

}  // namespace framework
}  // namespace paddle
