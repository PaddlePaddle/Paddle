/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jit/gen/seqpool.h"

#include "paddle/fluid/operators/jit/gen/act.h"  // for exp_float_consts ones
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void SeqPoolJitCode::genCode() {
  constexpr int block = YMM_FLOAT_BLOCK;
  constexpr int max_num_regs = 8;
  const int num_block = w_ / block;
  const int num_groups = num_block / max_num_regs;
  int rest_num_regs = num_block % max_num_regs;
  mov(reg32_int_h, dword[param_attr]);
  if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
    mov(reg_tmp, reinterpret_cast<size_t>(exp_float_consts));
    vmovups(xmm_t(1), ptr[reg_tmp + OFFSET_EXP_ONE]);
    mov(reg_tmp, reinterpret_cast<size_t>(fp_h_));
    fild(dword[param_attr]);
    fstp(dword[reg_tmp]);
    vmovss(xmm_t(0), ptr[reg_tmp]);
    if (type_ == SeqPoolType::kSqrt) {
      vsqrtps(xmm_t(0), xmm_t(0));
    }
    vdivps(xmm_t(1), xmm_t(1), xmm_t(0));
    vmovss(ptr[reg_tmp], xmm_t(1));
  }
  const int group_len = max_num_regs * block * sizeof(float);
  for (int g = 0; g < num_groups; ++g) {
    pool_height<ymm_t>(g * group_len, block, max_num_regs);
  }
  if (rest_num_regs > 0) {
    pool_height<ymm_t>(num_groups * group_len, block, rest_num_regs);
  }
  // part of rest_w * height
  const int rest = w_ % block;
  pool_height_of_rest_width(rest, (w_ - rest) * sizeof(float), max_num_regs);
  ret();
}

class SeqPoolCreator : public JitCodeCreator<seq_pool_attr_t> {
 public:
  bool CanBeUsed(const seq_pool_attr_t& attr) const override {
    return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
  }
  size_t CodeSize(const seq_pool_attr_t& attr) const override {
    return 96 + ((attr.w / YMM_FLOAT_BLOCK + 4 /* for rest */) *
                     4 /* load, mul and save */
                 + 256) *
                    16;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const seq_pool_attr_t& attr) const override {
    PADDLE_ENFORCE_GT(attr.w,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute width of SeqPool should "
                          "be larger than 0. But it is %d.",
                          attr.w));
    PADDLE_ENFORCE_GT(attr.h,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute height of SeqPool should "
                          "be larger than 0. But it is %d.",
                          attr.h));
    return make_unique<SeqPoolJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kSeqPool, gen::SeqPoolCreator);
