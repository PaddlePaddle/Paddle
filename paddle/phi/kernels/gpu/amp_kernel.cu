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

#include "paddle/phi/kernels/amp_kernel.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/memory.h"

namespace phi {

template <typename T>
__global__ void InverseAndMemset(const T* s, T* o, bool* found_inf) {
  *o = 1.0 / *s;
  *found_inf = false;
}

template <typename T, typename MT>
__global__ void CheckFiniteAndUnscale(const T** xs,
                                      const MT* scale,
                                      int64_t size,
                                      int64_t* starts,
                                      bool* found_inf,
                                      T** outs) {
  const int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  // copy starts array from global memory to shared memory
  extern __shared__ int64_t s_starts[];
  for (int i = threadIdx.x; i <= size; i += blockDim.x) {
    s_starts[i] = starts[i];
  }
  __syncthreads();

  const int64_t num = s_starts[size];
  int xs_index = 0;
  bool local_found_inf = false;
  const MT local_scale = *scale;
  for (int64_t idx = tid; idx < num; idx += gridDim.x * blockDim.x) {
    // get the "out" index of "id"
    // For example:
    // idx = 15, starts = [0, 10, 10, 20, 30]
    // because 10 <= idx < 20 ==>
    // the idx element locate in the 3rd tensor (notice the 2nd tensor size is
    // 0)
    int next_xs_index = xs_index;
    while (idx >= s_starts[next_xs_index]) next_xs_index++;
    xs_index = next_xs_index - 1;

    // get in data and out data
    const T* in = xs[xs_index];
    T* out = outs[xs_index];
    int64_t in_idx = idx - s_starts[xs_index];

    // Unscale
    MT val = static_cast<MT>(in[in_idx]) * local_scale;
    T narrow_val = static_cast<T>(val);
    out[in_idx] = narrow_val;

    // CheckFinite
    if (!isfinite(narrow_val)) {
      local_found_inf = true;
    }
  }
  if (local_found_inf) {
    *found_inf = true;
  }
}

template <typename T, typename Context>
void CheckFiniteAndUnscaleKernel(const Context& dev_ctx,
                                 const std::vector<const DenseTensor*>& xs,
                                 const DenseTensor& scale,
                                 std::vector<DenseTensor*> outs,
                                 DenseTensor* found_infinite) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  const MPDType* scale_data = scale.data<MPDType>();
  bool* found_inf_data = dev_ctx.template Alloc<bool>(found_infinite);

  DenseTensor inverse_scale = Empty<MPDType>(dev_ctx, {1});
  MPDType* inverse_scale_v = inverse_scale.template data<MPDType>();

  InverseAndMemset<MPDType><<<1, 1, 0, dev_ctx.stream()>>>(
      scale_data, inverse_scale_v, found_inf_data);

  size_t xs_size = xs.size();
  if (xs_size == 0) return;

  const auto& cpu_place = phi::CPUPlace();
  // calculate each tensor's start index and copy to device
  auto h_starts_tensor =
      paddle::memory::Alloc(cpu_place, (xs_size + 1) * sizeof(int64_t));
  int64_t* h_starts = reinterpret_cast<int64_t*>(h_starts_tensor->ptr());

  auto d_starts_tensor =
      paddle::memory::Alloc(dev_ctx, (xs_size + 1) * sizeof(int64_t));
  int64_t* d_starts = reinterpret_cast<int64_t*>(d_starts_tensor->ptr());

  // the start index value of each tensor is
  // the sum of previous tensor's size. For example:
  // x = [10, 0, 10, 10] ==> starts = [0, 10, 10, 20, 30]
  h_starts[0] = 0;
  for (int i = 1; i <= xs_size; i++) {
    h_starts[i] = h_starts[i - 1] + xs[i - 1]->numel();
  }
  int64_t total_num = h_starts[xs_size];
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       d_starts,
                       cpu_place,
                       h_starts,
                       (xs_size + 1) * sizeof(int64_t),
                       dev_ctx.stream());

  // copy each tensor's data address to device
  auto h_mem = paddle::memory::Alloc(cpu_place, 2 * xs_size * sizeof(T*));
  const T** h_xs = reinterpret_cast<const T**>(h_mem->ptr());
  T** h_outs = reinterpret_cast<T**>(h_mem->ptr()) + xs_size;

  auto d_mem = paddle::memory::Alloc(dev_ctx, 2 * xs_size * sizeof(T*));
  const T** d_xs = reinterpret_cast<const T**>(d_mem->ptr());
  T** d_outs = reinterpret_cast<T**>(d_mem->ptr()) + xs_size;

  for (size_t i = 0; i < xs_size; ++i) {
    h_xs[i] = xs[i]->data<T>();
    h_outs[i] = dev_ctx.template Alloc<T>(outs[i]);
  }
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       d_xs,
                       cpu_place,
                       h_xs,
                       2 * xs_size * sizeof(T*),
                       dev_ctx.stream());

  // Launch Kernel
  int threads_per_block = std::min(static_cast<int64_t>(1024), total_num);
  int elements_per_block =
      threads_per_block * 20;  // each thread deal with 20 number
  int blocks_per_grid =
      (total_num + elements_per_block - 1) / elements_per_block;
  CheckFiniteAndUnscale<T, MPDType><<<blocks_per_grid,
                                      threads_per_block,
                                      (xs_size + 1) * sizeof(int64_t),
                                      dev_ctx.stream()>>>(
      d_xs, inverse_scale_v, xs_size, d_starts, found_inf_data, d_outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(check_finite_and_unscale,
                   GPU,
                   ALL_LAYOUT,
                   phi::CheckFiniteAndUnscaleKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
