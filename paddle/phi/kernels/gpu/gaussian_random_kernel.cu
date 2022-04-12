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

#include <thrust/random.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"

#include "paddle/fluid/framework/generator.h"

namespace phi {

template <typename T>
struct GaussianGenerator {
  T mean_, std_;
  unsigned int seed_;
  unsigned int offset_ = 0;

  __host__ __device__ GaussianGenerator(T mean, T std, int seed)
      : mean_(mean), std_(std), seed_(seed) {}

  __host__ __device__ GaussianGenerator(T mean, T std, int seed, int offset)
      : mean_(mean), std_(std), seed_(seed), offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    thrust::normal_distribution<MT> dist(static_cast<MT>(mean_),
                                         static_cast<MT>(std_));
    unsigned int new_n = n + offset_;
    rng.discard(new_n);
    MT out = dist(rng);
    return static_cast<T>(out);
  }
};

template <typename T, typename Context>
void GaussianRandomKernel(const Context& dev_ctx,
                          const IntArray& shape,
                          float mean,
                          float std,
                          int seed,
                          DataType dtype,
                          DenseTensor* out) {
  auto tensor = out;

  bool seed_flag = false;
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    seed_flag = true;
  }

  tensor->Resize(phi::make_ddim(shape.GetData()));

  T* data = dev_ctx.template Alloc<T>(tensor);

  int64_t size = tensor->numel();

  int device_id = dev_ctx.GetPlace().GetDeviceId();
  auto gen_cuda = paddle::framework::GetDefaultCUDAGenerator(device_id);

  if (gen_cuda->GetIsInitPy() && seed_flag) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    funcs::normal_distribution<MT> dist;
    funcs::normal_transform<MT> trans(static_cast<MT>(mean),
                                      static_cast<MT>(std));
    funcs::distribution_and_transform<T>(dev_ctx, tensor, dist, trans);
  } else {
    auto func =
        GaussianGenerator<T>(static_cast<T>(mean), static_cast<T>(std), seed);
    IndexKernel<T, GaussianGenerator<T>>(dev_ctx, tensor, func);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian_random,
                   GPU,
                   ALL_LAYOUT,
                   phi::GaussianRandomKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
