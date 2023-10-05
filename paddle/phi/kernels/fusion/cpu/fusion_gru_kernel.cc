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

#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"
#include "paddle/phi/kernels/funcs/jit/kernels.h"
#include "paddle/phi/kernels/funcs/sequence2batch.h"

namespace phi {
namespace fusion {

#define INIT_BASE_DEFINES                                  \
  auto x_lod = x.lod();                                    \
  auto x_dims = x.dims(); /* T x M*/                       \
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1) \
                        ? phi::flatten_to_2d(x_dims, 1)    \
                        : x_dims;                          \
  auto wh_dims = weight_h.dims(); /* D x 3D*/              \
  const int total_T = x_mat_dims[0];                       \
  const int D3 = wh_dims[1]

#define INIT_OTHER_DEFINES                                                   \
  const int M = x_mat_dims[1];                                               \
  const int D = wh_dims[0];                                                  \
  const int D2 = D * 2;                                                      \
  const phi::jit::gru_attr_t attr(D,                                         \
                                  phi::jit::to_kerneltype(gate_activation),  \
                                  phi::jit::to_kerneltype(activation));      \
  phi::jit::gru_t one_step;                                                  \
  auto ComputeH1 =                                                           \
      phi::jit::KernelFuncs<phi::jit::GRUH1Tuple<T>, phi::CPUPlace>::Cache() \
          .At(attr);                                                         \
  auto ComputeHtPart1 = phi::jit::KernelFuncs<phi::jit::GRUHtPart1Tuple<T>,  \
                                              phi::CPUPlace>::Cache()        \
                            .At(attr);                                       \
  auto ComputeHtPart2 = phi::jit::KernelFuncs<phi::jit::GRUHtPart2Tuple<T>,  \
                                              phi::CPUPlace>::Cache()        \
                            .At(attr);                                       \
  const T* x_data = x.data<T>();                                             \
  const T* wx_data = weight_x.data<T>();                                     \
  const T* wh_data = weight_h.data<T>();                                     \
  T* xx_data = dev_ctx.template Alloc<T>(xx)

template <typename T, typename Context>
void FusionGRUKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& h0,
                     const DenseTensor& weight_x,
                     const DenseTensor& weight_h,
                     const paddle::optional<DenseTensor>& bias,
                     const DenseTensor& recorder_h0,
                     const std::string& activation,
                     const std::string& gate_activation,
                     const bool is_reverse,
                     const bool use_seq,
                     const bool origin_mode,
                     const bool use_mkldnn,
                     const std::string& mkldnn_data_type,
                     const float scale_data,
                     const float shift_data,
                     const std::vector<float>& scale_weights,
                     const bool force_fp32_output,
                     DenseTensor* xx,
                     DenseTensor* batched_input,
                     DenseTensor* batched_out,
                     DenseTensor* hidden) {
  if (use_seq) {
    SeqCompute();
  } else {
    BatchCompute();
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    fusion_gru, CPU, ALL_LAYOUT, phi::fusion::FusionGRUKernel, float, double) {}
