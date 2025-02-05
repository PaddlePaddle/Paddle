// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifndef _MSC_VER
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#endif

#include "paddle/common/errors.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"

#include "paddle/phi/kernels/gpu/shuffle_batch_utils.h"
#include "paddle/phi/kernels/shuffle_batch_kernel.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/funcs/shuffle_batch.cu.h"
#endif

namespace phi {

template <typename T, typename Context>
void ShuffleBatchKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& seed,
                        int startup_seed,
                        DenseTensor* out,
                        DenseTensor* shuffleidx,
                        DenseTensor* seed_out) {
#ifdef _MSC_VER
  PADDLE_THROW(common::errors::Unimplemented(
      "GPU shuffle_batch is not supported on Windows yet"));
#else
  int64_t x_embed_size = x.dims()[x.dims().size() - 1];
  int64_t elem_size = 1;
  for (int i = 0; i < x.dims().size() - 1; i++) {
    elem_size *= x.dims()[i];
  }
  shuffleidx->Resize(common::make_ddim({elem_size}));

  int64_t seed_int = 0;
  if (seed.initialized()) {
    const auto& seed_place = seed.place().GetType();
    bool is_gpu_place = seed_place == phi::AllocationType::GPU;
    if (is_gpu_place) {
      // NOTE: We have overwritten GetKernelTypeForVar, so seed_place would
      // not be CUDAPlace in practice. This case would only happen in Python
      // op_test framework.
      phi::DenseTensor tmp_tensor;
      phi::Copy(dev_ctx, seed, phi::CPUPlace(), false, &tmp_tensor);
      seed_int = *(tmp_tensor.data<int64_t>());
    } else {
      seed_int = *(seed.data<int64_t>());
    }
  } else {
    seed_int = startup_seed;
  }

  auto* shuffleidx_data = dev_ctx.template Alloc<int64_t>(shuffleidx);

#ifdef PADDLE_WITH_CUDA
  // CacheAllocator allocator(dev_ctx.GetPlace());
  phi::memory_utils::ThrustAllocator<cudaStream_t> allocator(dev_ctx.GetPlace(),
                                                             dev_ctx.stream());
  const auto& exec_policy = thrust::cuda::par(allocator).on(dev_ctx.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
  thrust::random::default_random_engine engine(seed_int);
  thrust::counting_iterator<int64_t> cnt_iter(0);
#ifdef PADDLE_WITH_CUDA
  phi::funcs::shuffle_copy_fixed(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec_policy)),
#else
  thrust::shuffle_copy(exec_policy,
#endif
      cnt_iter,
      cnt_iter + elem_size,
      thrust::device_pointer_cast(shuffleidx_data),
      engine);
  // TODO(zengjinle): for small data, direct cudaMemcpy may be better
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  ReorderFunctor<T, true> functor(
      x_data, shuffleidx_data, out_data, x_embed_size);
  phi::funcs::ForRange<phi::GPUContext> for_range(dev_ctx,
                                                  elem_size * x_embed_size);
  for_range(functor);
  seed_out->Resize(common::make_ddim({1}));
  auto* seed_out_data = dev_ctx.template HostAlloc<int64_t>(seed_out);
  *seed_out_data = engine();
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(shuffle_batch,
                   GPU,
                   ALL_LAYOUT,
                   phi::ShuffleBatchKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT64);
}
#endif
