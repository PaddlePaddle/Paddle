/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <string>
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/kernels/strings/strings_copy_kernel.h"
#include "paddle/phi/kernels/strings/strings_empty_kernel.h"

namespace phi {
namespace tests {

using DDim = phi::DDim;
using pstring = phi::dtype::pstring;

TEST(DEV_API, strings_copy) {
  // 1. create tensor
  const DDim dims({2, 3});
  StringTensorMeta meta(dims);

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();

  auto* dev_ctx = reinterpret_cast<phi::CPUContext*>(pool.Get(phi::CPUPlace()));
  auto* gpu_dev_ctx =
      reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  StringTensor string_src = phi::strings::Empty(*dev_ctx, std::move(meta));
  StringTensor string_dst = phi::strings::Empty(*dev_ctx, std::move(meta));
  // 2. Assign input text
  const char* input[] = {"A Short Pstring.",
                         "A Large Pstring Whose Length Is Longer Than 22.",
                         "abc",
                         "defg",
                         "hijklmn",
                         "opqrst"};
  pstring* string_src_data = dev_ctx->template Alloc<pstring>(&string_src);
  for (int i = 0; i < string_src.numel(); ++i) {
    string_src_data[i] = input[i];
  }
  StringTensor string_gpu1 = phi::strings::Empty(*gpu_dev_ctx, std::move(meta));
  StringTensor string_gpu2 = phi::strings::Empty(*gpu_dev_ctx, std::move(meta));

  // cpu->gpu
  phi::strings::Copy(*gpu_dev_ctx, string_src, false, &string_gpu1);
  // gpu->gpu
  phi::strings::Copy(*gpu_dev_ctx, string_gpu1, false, &string_gpu2);
  // gpu->cpu
  phi::strings::Copy(*gpu_dev_ctx, string_gpu2, false, &string_dst);
  for (int64_t i = 0; i < string_src.numel(); i++) {
    ASSERT_EQ(string_src.data()[i], string_dst.data()[i]);
  }
}

}  // namespace tests
}  // namespace phi
