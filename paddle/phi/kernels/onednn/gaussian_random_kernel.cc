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

#include "paddle/phi/kernels/gaussian_random_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GaussianRandomKernel(const Context& ctx,
                          const IntArray& shape,
                          float mean,
                          float std,
                          int seed,
                          DataType dtype,
                          DenseTensor* out) {
  std::normal_distribution<T> dist(mean, std);
  auto engine = std::make_shared<std::mt19937_64>();
  engine->seed(seed);

  T* data = ctx.template Alloc<T>(out);
  for (int64_t i = 0; i < out->numel(); ++i) {
    data[i] = dist(*engine);
  }

  out->Resize(phi::make_ddim(shape.GetData()));
  dnnl::memory::desc out_mem_desc(
      vectorize(out->dims()),
      funcs::ToOneDNNDataType(out->dtype()),
      funcs::GetPlainOneDNNFormat(out->dims().size()));
  out->set_mem_desc(out_mem_desc);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gaussian_random, OneDNN, ALL_LAYOUT, phi::GaussianRandomKernel, float) {}
