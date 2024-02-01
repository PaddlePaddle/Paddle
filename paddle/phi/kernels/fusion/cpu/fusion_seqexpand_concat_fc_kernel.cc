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

#include <string>

#include "paddle/common/errors.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/cpu_vec.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusionSeqExpandConcatFCKernel(const Context& dev_ctx,
                                   const std::vector<const DenseTensor*>& x,
                                   const DenseTensor& fc_weight,
                                   const paddle::optional<DenseTensor>& fc_bias,
                                   const std::string& fc_activation,
                                   DenseTensor* out,
                                   DenseTensor* fc_out) {
  auto* ref_in = x[0];
  auto ref_lod = ref_in->lod();
  auto in1_lod = x[1]->lod();
  auto ref_dims = ref_in->dims();  // T x M0
  auto in1_dims = x[1]->dims();    // N x M1
  auto w_dims = fc_weight.dims();
  const int N = static_cast<int>(ref_lod[0].size() - 1);
  const int total_T = static_cast<int>(ref_dims[0]);
  const int M0 = static_cast<int>(ref_dims[1]);
  const int M1 = static_cast<int>(in1_dims[1]);
  const int D = static_cast<int>(w_dims[1]);

  // some check and fcout should be reshape here
  // since infershape can not get lod info
  PADDLE_ENFORCE_EQ(
      ref_lod.size(),
      1UL,
      phi::errors::InvalidArgument(
          "Only support input lod size is 1, but received value is: %d.",
          ref_lod.size()));
  PADDLE_ENFORCE_EQ(
      in1_lod.size(),
      1UL,
      phi::errors::InvalidArgument(
          "Only support input lod size is 1, but received value is: %d.",
          in1_lod.size()));
  PADDLE_ENFORCE_EQ(static_cast<int>(in1_lod[0].size() - 1),
                    N,
                    phi::errors::InvalidArgument(
                        "Batch size of all inputs should be equal to %d, but "
                        "received value is: %d.",
                        N,
                        static_cast<int>(in1_lod[0].size() - 1)));
  PADDLE_ENFORCE_EQ(
      static_cast<int>(in1_lod[0][N]),
      N,
      phi::errors::InvalidArgument("Seq_length of other inputs should "
                                   "be %d, but received value is: %d.",
                                   N,
                                   static_cast<int>(in1_lod[0][N])));
  PADDLE_ENFORCE_EQ(
      in1_dims[0],
      N,
      phi::errors::InvalidArgument(
          "input height should be batch size: %d, but received value is %d.",
          N,
          in1_dims[0]));
  for (size_t i = 2; i < x.size(); ++i) {
    PADDLE_ENFORCE_EQ(x[i]->dims()[0],
                      N,
                      phi::errors::InvalidArgument(
                          "All other inputs height should be equal to %d, "
                          "but received value is: %d.",
                          N,
                          x[i]->dims()[0]));
    PADDLE_ENFORCE_EQ(x[i]->lod(),
                      in1_lod,
                      phi::errors::InvalidArgument(
                          "All other inputs should have same lod: %d, but "
                          "received value is: %d.",
                          in1_lod,
                          x[i]->lod()));
  }
  fc_out->Resize({N, D});

  std::function<void(const int, const T*, T*)> fc_act;
  if (phi::backends::cpu::MayIUse(phi::backends::cpu::avx)) {
    phi::funcs::VecActivations<T, phi::backends::cpu::avx> act_functor;
    fc_act = act_functor(fc_activation);
  } else {
    phi::funcs::VecActivations<T, phi::backends::cpu::isa_any> act_functor;
    fc_act = act_functor(fc_activation);
  }

  const T* ref_in_data = ref_in->data<T>();
  const T* in1_data = x[1]->data<T>();
  const T* w_data = fc_weight.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  T* fc_out_data = dev_ctx.template Alloc<T>(fc_out);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  phi::funcs::FCFunctor<Context, T> fc;
  fc(dev_ctx,
     total_T,
     D,
     M0,
     ref_in_data,
     w_data,
     out_data,
     fc_bias ? fc_bias->data<T>() : nullptr);
  w_data = w_data + M0 * D;
  // first write on
  blas.MatMul(N, D, M1, in1_data, w_data, fc_out_data);
  w_data = w_data + M1 * D;
  for (size_t i = 2; i < x.size(); ++i) {
    // add on
    const T* in_data = x[i]->data<T>();
    const int K = static_cast<int>(x[i]->dims()[1]);
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              N,
              D,
              K,
              static_cast<T>(1),
              in_data,
              K,
              w_data,
              D,
              static_cast<T>(1),
              fc_out_data,
              D);
    w_data = w_data + K * D;
  }
  T* cur_out_data = out_data;
  for (int i = 0; i < N; ++i) {
    int seq_len = static_cast<int>(ref_lod[0][i + 1] - ref_lod[0][i]);
    T* src = fc_out_data + i * D;
    for (int step = 0; step < seq_len; ++step) {
      blas.VADD(D, cur_out_data, src, cur_out_data);
      cur_out_data = cur_out_data + D;
    }
  }
  fc_act(total_T * D, out_data, out_data);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fusion_seqexpand_concat_fc,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusionSeqExpandConcatFCKernel,
                   float,
                   double) {}
