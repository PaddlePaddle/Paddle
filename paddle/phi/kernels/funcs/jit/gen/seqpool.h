/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/jit/gen/jitcode.h"

namespace phi {
namespace jit {
namespace gen {

class SeqPoolJitCode : public JitCode {
 public:
  explicit SeqPoolJitCode(const seq_pool_attr_t& attr,
                          size_t code_size = 256 * 1024,
                          void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), w_(attr.w), type_(attr.type) {
    if (!(type_ == SeqPoolType::kSum || type_ == SeqPoolType::kAvg ||
          type_ == SeqPoolType::kSqrt)) {
      PADDLE_THROW(common::errors::Unimplemented(
          "Only supports sum, average and sqrt pool type."));
    }
    fp_h_[0] = 1.f;
    this->genCode();
  }

  std::string name() const override {
    std::string base = "SeqPoolJitCode";
    if (type_ == SeqPoolType::kSum) {
      base += "_Sum";
    } else if (type_ == SeqPoolType::kAvg) {
      base += "_Avg";
    } else if (type_ == SeqPoolType::kSqrt) {
      base += "_Sqrt";
    }
    base += ("_W" + std::to_string(w_));
    return base;
  }
  void genCode() override;

 protected:
  template <typename JMM>
  void pool_height(int w_offset, int block, int max_num_regs) {
    int offset = w_offset;
    for (int i = 0; i < max_num_regs; ++i) {
      vmovups(JMM(i), ptr[param_src + offset]);
      offset += sizeof(float) * block;
    }
    cmp(reg32_int_h, 1);
    Label l_next_h, l_h_done;
    jle(l_h_done, T_NEAR);
    mov(reg_h_i, 1);
    mov(reg_tmp, param_src);
    add(reg_tmp, w_ * sizeof(float) + w_offset);
    L(l_next_h);
    {
      mov(reg_ptr_src_i, reg_tmp);
      for (int i = 0; i < max_num_regs; ++i) {
        vmovups(JMM(i + max_num_regs), ptr[reg_ptr_src_i]);
        // sum anyway
        vaddps(JMM(i), JMM(i), JMM(i + max_num_regs));
        add(reg_ptr_src_i, sizeof(float) * block);
      }
      inc(reg_h_i);
      add(reg_tmp, w_ * sizeof(float));
      cmp(reg_h_i, reg32_int_h);
      jl(l_next_h, T_NEAR);
    }
    L(l_h_done);
    // save right now
    if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
      mov(reg_tmp, reinterpret_cast<size_t>(fp_h_));
      vbroadcastss(JMM(max_num_regs), ptr[reg_tmp]);
    }
    offset = w_offset;
    for (int i = 0; i < max_num_regs; ++i) {
      if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
        vmulps(JMM(i), JMM(i), JMM(max_num_regs));
      }
      vmovups(ptr[param_dst + offset], JMM(i));
      offset += sizeof(float) * block;
    }
  }

  void pool_height_of_rest_width(int rest, int w_offset, int max_num_regs) {
    const int rest_used_num_regs = load_rest(rest, w_offset, 0);
    const bool has_block4 = rest / 4 > 0;
    const bool has_block2 = (rest % 4) / 2 > 0;
    const bool has_block1 = (rest % 2) == 1;
    cmp(reg32_int_h, 1);
    Label l_next_h, l_h_done;
    jle(l_h_done, T_NEAR);
    mov(reg_h_i, 1);
    mov(reg_tmp, param_src);
    add(reg_tmp, w_ * sizeof(float) + w_offset);
    L(l_next_h);
    {
      int reg_idx = 0;
      mov(reg_ptr_src_i, reg_tmp);
      if (has_block4) {
        vmovups(xmm_t(reg_idx + max_num_regs), ptr[reg_ptr_src_i]);
        add(reg_ptr_src_i, sizeof(float) * 4);
        reg_idx++;
      }
      if (has_block2) {
        vmovups(xmm_t(reg_idx + max_num_regs), ptr[reg_ptr_src_i]);
        add(reg_ptr_src_i, sizeof(float) * 2);
        reg_idx++;
      }
      if (has_block1) {
        vmovss(xmm_t(reg_idx + max_num_regs), ptr[reg_ptr_src_i]);
        reg_idx++;
      }
      PADDLE_ENFORCE_EQ(
          reg_idx,
          rest_used_num_regs,
          common::errors::InvalidArgument(
              "All heights of SeqPool should use the same number of registers."
              "It equals to the numbr of rest registers. But use %d registers "
              "and the numbr of rest registers is %d.",
              reg_idx,
              rest_used_num_regs));
      for (int i = 0; i < reg_idx; ++i) {
        vaddps(xmm_t(i), xmm_t(i), xmm_t(i + max_num_regs));
      }
      inc(reg_h_i);
      add(reg_tmp, w_ * sizeof(float));
      cmp(reg_h_i, reg32_int_h);
      jl(l_next_h, T_NEAR);
    }
    L(l_h_done);
    // save right now
    if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
      mov(reg_tmp, reinterpret_cast<size_t>(fp_h_));
      vbroadcastss(xmm_t(max_num_regs), ptr[reg_tmp]);
      for (int i = 0; i < rest_used_num_regs; ++i) {
        vmulps(xmm_t(i), xmm_t(i), xmm_t(max_num_regs));
      }
    }
    save_rest(rest, w_offset);
  }

  // return the number of used regs, use start from reg 0
  int load_rest(int rest,
                int w_offset,
                const int num_shift_regs,
                const int reg_start = 0) {
    const bool has_block4 = rest / 4 > 0;
    const bool has_block2 = (rest % 4) / 2 > 0;
    const bool has_block1 = (rest % 2) == 1;
    int reg_idx = reg_start;
    if (has_block4) {
      vmovups(xmm_t(reg_idx + num_shift_regs), ptr[param_src + w_offset]);
      w_offset += sizeof(float) * 4;
      reg_idx++;
    }
    if (has_block2) {
      vmovq(xmm_t(reg_idx + num_shift_regs), ptr[param_src + w_offset]);
      w_offset += sizeof(float) * 2;
      reg_idx++;
    }
    if (has_block1) {
      vmovss(xmm_t(reg_idx + num_shift_regs), ptr[param_src + w_offset]);
      reg_idx++;
    }
    return reg_idx;
  }

  // use reg start from 0
  void save_rest(int rest, int w_offset, int reg_start = 0) {
    const bool has_block4 = rest / 4 > 0;
    const bool has_block2 = (rest % 4) / 2 > 0;
    const bool has_block1 = (rest % 2) == 1;
    int reg_idx = reg_start;
    if (has_block4) {
      vmovups(ptr[param_dst + w_offset], xmm_t(reg_idx));
      w_offset += sizeof(float) * 4;
      reg_idx++;
    }
    if (has_block2) {
      vmovq(ptr[param_dst + w_offset], xmm_t(reg_idx));
      w_offset += sizeof(float) * 2;
      reg_idx++;
    }
    if (has_block1) {
      vmovss(ptr[param_dst + w_offset], xmm_t(reg_idx));
    }
  }

 private:
  float ALIGN32_BEG fp_h_[1] ALIGN32_END;
  int w_;
  SeqPoolType type_;
  reg64_t param_src{abi_param1};
  reg64_t param_dst{abi_param2};
  reg64_t param_attr{abi_param3};
  reg64_t reg_tmp{rax};

  reg32_t reg32_int_h{r8d};
  reg32_t reg32_fp_h{r9d};

  reg64_t reg_h_i{r10};
  reg64_t reg_ptr_src_i{r11};
};

}  // namespace gen
}  // namespace jit
}  // namespace phi
