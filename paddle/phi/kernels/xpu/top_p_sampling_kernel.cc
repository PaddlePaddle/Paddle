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

PHI_DEFINE_EXPORTED_bool(xpu_top_p_sampling_use_fp16,
                         false,
                         "use fp16 to improve the inference performance of "
                         "top_p_sampling xpu kernel");
PHI_DEFINE_EXPORTED_bool(
    xpu_top_p_sampling_heuristic_threshold,
    20,
    "threshold of heuristic method used for xpu_top_p_sampling, default 20; if "
    "heuristic_threshold = -1, xpu_top_p_sampling don't use heuristic method, "
    "and will fallback to normal top_p_sampling; if heuristic_threshold > 0, "
    "xpu_top_p_sampling will enable heuristic method");

namespace phi {

template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& ps,
                        const paddle::optional<DenseTensor>& threshold,
                        const paddle::optional<DenseTensor>& topp_seed,
                        int random_seed,
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
  int bs = x_dims[0];
  int vocab_size = x_dims[1];
  int p_num = ps.numel();

  PADDLE_ENFORCE_EQ(
      p_num,
      bs,
      common::errors::PreconditionNotMet(
          "Expected bs == p_num, but got bs=%d, p_num=%d.", bs, p_num));

  std::vector<int64_t> infer_seed(bs, random_seed);
  if (topp_seed.get_ptr() != nullptr) {
    phi::TensorToVector(*topp_seed, dev_ctx, &infer_seed);
  }

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::vector<float> rand_coeff_cpu;
  for (int i = 0; i < bs; i++) {
    if (infer_seed[i] == -1) {
      std::shared_ptr<std::mt19937_64> engine =
          dev_ctx.GetGenerator()->GetCPUEngine();
      rand_coeff_cpu.push_back(dist(*engine));
    } else {
      std::mt19937_64 engine(infer_seed[i]);
      rand_coeff_cpu.push_back(dist(engine));
    }
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  float* rand_coeff_xpu = RAII_GUARD.alloc<float>(rand_coeff_cpu.size());
  int* ids_int_ptr = RAII_GUARD.alloc<int>(ids->numel());

  int r = xpu::do_host2device(dev_ctx.x_context(),
                              rand_coeff_cpu.data(),
                              rand_coeff_xpu,
                              rand_coeff_cpu.size() * sizeof(float));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "do_host2device");
  int heuristic_threshold = FLAGS_xpu_top_p_sampling_heuristic_threshold;

  if ((!FLAGS_xpu_top_p_sampling_use_fp16) ||
      std::is_same<T, phi::dtype::float16>::value) {
    r = xpu::faster_top_p_sampling<XPUType, int>(dev_ctx.x_context(),
                                                 x_ptr,
                                                 ps_ptr,
                                                 rand_coeff_xpu,
                                                 ids_int_ptr,
                                                 bs,
                                                 vocab_size,
                                                 out_ptr,
                                                 nullptr,
                                                 heuristic_threshold);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "top_p_sampling");
  } else {
    using XPUFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
    XPUFP16* x_fp16_ptr = RAII_GUARD.alloc<XPUFP16>(x.numel());
    XPUFP16* ps_fp16_ptr = RAII_GUARD.alloc<XPUFP16>(ps.numel());
    XPUFP16* out_fp16_ptr = RAII_GUARD.alloc<XPUFP16>(out->numel());

    float fp16_scale = 32768.f;  // experience value
    r = xpu::scale_cast_fusion<XPUType, XPUFP16>(
        dev_ctx.x_context(), x_ptr, x_fp16_ptr, x.numel(), fp16_scale);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale_cast_fusion");
    r = xpu::scale_cast_fusion<XPUType, XPUFP16>(
        dev_ctx.x_context(), ps_ptr, ps_fp16_ptr, ps.numel(), fp16_scale);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale_cast_fusion");

    r = xpu::faster_top_p_sampling<XPUFP16, int>(dev_ctx.x_context(),
                                                 x_fp16_ptr,
                                                 ps_fp16_ptr,
                                                 rand_coeff_xpu,
                                                 ids_int_ptr,
                                                 bs,
                                                 vocab_size,
                                                 out_fp16_ptr,
                                                 nullptr,
                                                 heuristic_threshold);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "top_p_sampling");

    r = xpu::scale_cast_fusion<XPUFP16, XPUType>(dev_ctx.x_context(),
                                                 out_fp16_ptr,
                                                 out_ptr,
                                                 out->numel(),
                                                 1.f / fp16_scale);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale_cast_fusion");
  }
  r = xpu::cast<int, int64_t>(
      dev_ctx.x_context(), ids_int_ptr, ids_ptr, ids->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
}

}  // namespace phi

PD_REGISTER_KERNEL(top_p_sampling,
                   XPU,
                   ALL_LAYOUT,
                   phi::TopPSamplingKernel,
                   float,
                   phi::dtype::float16) {}
