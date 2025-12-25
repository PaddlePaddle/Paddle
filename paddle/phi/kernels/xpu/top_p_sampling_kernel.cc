// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/top_p_sampling_kernel.h"
#include "xpu/refactor/customized_api.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/common/flags.h"

namespace phi {

template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& ps,
                        const paddle::optional<DenseTensor>& threshold,
                        const paddle::optional<DenseTensor>& topp_seed,
                        int64_t random_seed,
                        int k,
                        const std::string& mode,
                        DenseTensor* out,
                        DenseTensor* ids,
                        DenseTensor* topk_scores,
                        DenseTensor* topk_ids) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* ps_ptr = reinterpret_cast<const XPUType*>(ps.data<T>());
  XPUType* out_ptr = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));
  int64_t* ids_ptr = dev_ctx.template Alloc<int64_t>(ids);
  auto x_dims = x.dims();
  int64_t bs = x_dims[0];
  int64_t vocab_size = x_dims[1];

  XPUType* topk_scores_data = nullptr;
  int64_t* topk_ids_data = nullptr;
  if (k > 0) {
    topk_scores_data =
        reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(topk_scores));
    topk_ids_data = dev_ctx.template Alloc<int64_t>(topk_ids);
    int r = xpu::topk<XPUType, int64_t>(dev_ctx.x_context(),
                                        x_ptr,
                                        topk_scores_data,
                                        topk_ids_data,
                                        {bs, vocab_size},
                                        k,
                                        1,
                                        true,
                                        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::topk");
  }

  uint64_t seed_now = random_seed;
  uint64_t offset = 0;
  std::vector<int64_t> infer_seed(bs, random_seed);
  if (topp_seed.get_ptr() != nullptr) {
    phi::TensorToVector(*topp_seed, dev_ctx, &infer_seed);
    seed_now = infer_seed[0];
  } else {
    if (random_seed == -1) {
      auto gen_cuda = dev_ctx.GetGenerator();
      uint64_t increment = bs;
      auto seed_offset = gen_cuda->IncrementOffset(increment);
      seed_now = seed_offset.first;
      offset = seed_offset.second;
    }
  }

  DenseTensor k_threshold;
  k_threshold.Resize({bs});
  int* k_threshold_ptr = dev_ctx.template Alloc<int>(&k_threshold);
  if (threshold.get_ptr() != nullptr) {
    XPUType* threshold_data = reinterpret_cast<XPUType*>(
        const_cast<T*>(threshold.get_ptr()->data<T>()));
    // k_threshold = static_cast<int>((1 - infer_threshold[0]) * vocab_size)
    int r = xpu::scale<XPUType, float>(dev_ctx.x_context(),
                                       threshold_data,
                                       threshold_data,
                                       bs,
                                       true,
                                       -vocab_size,
                                       vocab_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::scale");
    r = xpu::cast<XPUType, int>(
        dev_ctx.x_context(), threshold_data, k_threshold_ptr, bs);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::cast");
  } else {
    int r = xpu::constant<int>(
        dev_ctx.x_context(), k_threshold_ptr, bs, vocab_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::constant");
  }

  DenseTensor ids_int;
  ids_int.Resize({bs});
  int* ids_int_ptr = dev_ctx.template Alloc<int>(&ids_int);
  int r =
      xpu::top_k_top_p_sampling_from_probs<XPUType, int>(dev_ctx.x_context(),
                                                         x_ptr,
                                                         k_threshold_ptr,
                                                         ps_ptr,
                                                         nullptr,
                                                         ids_int_ptr,
                                                         vocab_size,
                                                         1,
                                                         bs,
                                                         vocab_size,
                                                         true,
                                                         seed_now,
                                                         offset);
  PADDLE_ENFORCE_XDNN_SUCCESS(r,
                              "xpu::top_k_top_p_sampling_from_probs<XPUType");
  r = xpu::cast<int, int64_t>(dev_ctx.x_context(), ids_int_ptr, ids_ptr, bs);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::cast");
  r = xpu::constant<XPUType>(dev_ctx.x_context(), out_ptr, bs, 0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu::constant");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    top_p_sampling, XPU, ALL_LAYOUT, phi::TopPSamplingKernel, float) {}
