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

#include <vector>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/phi/backends/onednn/onednn_helper.h"
#endif

namespace paddle {
namespace framework {

struct CastDataLayout {
  CastDataLayout(const phi::DeviceContext* ctx,
                 const std::vector<int>& axis,
                 const phi::DenseTensor& in,
                 phi::DenseTensor* out)
      : in_(in), out_(out), ctx_(ctx), axis_(axis) {}

  const phi::DenseTensor in_;
  phi::DenseTensor* out_;
  const phi::DeviceContext* ctx_;
  const std::vector<int> axis_;

  template <typename T>
  void apply();
};

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to);

TEST_API void TransDataLayout(const phi::KernelKey& kernel_type_for_var,
                              const phi::KernelKey& expected_kernel_type,
                              const phi::DenseTensor& in,
                              phi::DenseTensor* out,
                              const phi::Place& place);

void TransDataLayout(phi::DataLayout from_layout,
                     phi::DataLayout to_layout,
                     phi::Place place,
                     const phi::DenseTensor& in,
                     phi::DenseTensor* out);

}  // namespace framework
}  // namespace paddle
