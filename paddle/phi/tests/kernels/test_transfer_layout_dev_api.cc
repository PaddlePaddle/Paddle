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

#include <gtest/gtest.h>

#include <memory>

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/transfer_layout_kernel.h"

namespace phi {
namespace tests {

#ifdef PADDLE_WITH_MKLDNN
TEST(DEV_API, transfer_layout) {
  // 1. create tensor

  const int n = 2;
  const int c = 3;
  const int h = 4;
  const int w = 5;

  DenseTensor x;
  MetaTensor meta_x(&x);
  meta_x.set_dtype(DataType::FLOAT32);
  meta_x.set_layout(DataLayout::ONEDNN);
  meta_x.set_dims(make_ddim({n, c, h, w}));

  DenseTensor out;

  // 2. test API
  auto& pool = phi::DeviceContextPool::Instance();
  auto place = phi::CPUPlace();
  auto* dev_ctx = static_cast<const phi::CPUContext*>(pool.GetByPlace(place));

  MetaTensor meta_out(&out);
  TransferLayoutInferMeta(x,
                          static_cast<int>(x.layout()),
                          static_cast<int>(DataLayout::NHWC),
                          &meta_out);
  TransferLayoutKernel<CPUContext>(*dev_ctx,
                                   x,
                                   static_cast<int>(x.layout()),
                                   static_cast<int>(DataLayout::NHWC),
                                   &out);

  // 3. check result
  std::vector<int64_t> expect_shape = {12, 3};
  ASSERT_EQ(out.dims(), make_ddim({n, h, w, c}));
  ASSERT_EQ(out.dims().size(), 4);
  ASSERT_EQ(out.meta().dtype, DataType::FLOAT32);
  ASSERT_EQ(out.meta().layout, DataLayout::NHWC);
}

#endif
}  // namespace tests
}  // namespace phi
