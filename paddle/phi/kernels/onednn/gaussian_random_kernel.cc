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
#include "paddle/fluid/framework/generator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/onednn/onednn_reuse.h"

namespace phi {

template <typename T, typename Context>
void GaussianRandomKernel(const Context& ctx,
                          const IntArray& shape,
                          float mean,
                          float std,
                          int seed,
                          DataType dtype,
                          DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = out->mutable_data<T>(ctx.GetPlace());
  std::normal_distribution<T> dist(mean, std);
  auto engine = paddle::framework::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < out->numel(); ++i) {
    data[i] = dist(*engine);
  }

  dnnl::memory::desc out_mem_desc(
      phi::vectorize(out->dims()),
      paddle::framework::ToMKLDNNDataType(
          paddle::framework::TransToProtoVarType(out->dtype())),
      paddle::platform::GetPlainMKLDNNFormat(out->dims().size()));

  out->set_mem_desc(out_mem_desc);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gaussian_random, OneDNN, ALL_LAYOUT, phi::GaussianRandomKernel, float) {}
