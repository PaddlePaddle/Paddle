/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_macro.h"
#include "paddle/fluid/platform/enforce.h"

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

namespace jit = platform::jit;

/* LSTM JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class LSTMKernelImpl : public LSTMKernel<T> {
 public:
  explicit LSTMKernelImpl(int d, const std::string& act_gate,
                          const std::string& act_cand,
                          const std::string& act_cell)
      : LSTMKernel<T>() {
    d_ = d;
    d2_ = d * 2;
    d3_ = d * 3;
    auto GetActKernel = [&](const std::string& type,
                            int n) -> std::shared_ptr<const VActKernel<T>> {
      if (type == "sigmoid") {
        return std::dynamic_pointer_cast<const VActKernel<T>>(
            KernelPool::Instance().template Get<VSigmoidKernel<T>>(n));
      } else if (type == "relu") {
        return std::dynamic_pointer_cast<const VActKernel<T>>(
            KernelPool::Instance().template Get<VReluKernel<T>>(n));
      } else if (type == "tanh") {
        return std::dynamic_pointer_cast<const VActKernel<T>>(
            KernelPool::Instance().template Get<VTanhKernel<T>>(n));
      } else if (type == "identity" || type == "") {
        return std::dynamic_pointer_cast<const VActKernel<T>>(
            KernelPool::Instance().template Get<VIdentityKernel<T>>(n));
      }
      PADDLE_THROW("Not support type: %s", type);
    };
    act_gate_3d_ = GetActKernel(act_gate, d * 3);
    act_cand_d_ = GetActKernel(act_cand, d);
    act_cell_d_ = GetActKernel(act_cell, d);
    vmul_d_ = KernelPool::Instance().template Get<VMulKernel<T>>(d);
    vadd_d_ = KernelPool::Instance().template Get<VAddKernel<T>>(d);
  }

  void ComputeCtHt(T* gates, const T* ct_1, T* ct, T* ht) const override {
    // gates: W_ch, W_ih, W_fh, W_oh
    act_gate_3d_->Compute(gates + d_, gates + d_);

    /* C_t = C_t-1 * fgated + cand_gated * igated */
    act_cand_d_->Compute(gates, gates);
    vmul_d_->Compute(gates, gates + d_, gates + d_);
    vmul_d_->Compute(ct_1, gates + d2_, gates + d2_);
    vadd_d_->Compute(gates + d_, gates + d2_, ct);

    /* H_t = act_cell(C_t) * ogated */
    act_cell_d_->Compute(ct, gates + d2_);
    vmul_d_->Compute(gates + d2_, gates + d3_, ht);
  }

 private:
  int d_, d2_, d3_;
  std::shared_ptr<const VActKernel<T>> act_gate_3d_, act_cand_d_, act_cell_d_;
  std::shared_ptr<const VMulKernel<T>> vmul_d_;
  std::shared_ptr<const VAddKernel<T>> vadd_d_;
};

#define JITKERNEL_DECLARE_LSTM(ker_class, ker_dtype)                   \
  template <>                                                          \
  std::shared_ptr<const ker_class<ker_dtype>>                          \
  KernelPool::Get<ker_class<ker_dtype>, int, const std::string&,       \
                  const std::string&, const std::string&>(             \
      int d, const std::string& act_gate, const std::string& act_cand, \
      const std::string& act_cell)

#define JITKERNEL_KEY_LSTM(ker_key, dtype_key) \
  #ker_key #dtype_key + std::to_string(d) + act_gate + act_cand + act_cell

#define JITKERNEL_NEW_LSTM_IMPL(ker, dtype, isa, k)                     \
  p = std::dynamic_pointer_cast<ker<dtype>>(                            \
      std::make_shared<ker##Impl<dtype, isa, k>>(d, act_gate, act_cand, \
                                                 act_cell))

REGISTER_JITKERNEL_ARGS(lstm, LSTMKernel, JITKERNEL_DECLARE_LSTM,
                        JITKERNEL_KEY_LSTM, JITKERNEL_NEW_LSTM_IMPL);

#undef JITKERNEL_DECLARE_LSTM
#undef JITKERNEL_KEY_LSTM
#undef JITKERNEL_NEW_LSTM_IMPL

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
