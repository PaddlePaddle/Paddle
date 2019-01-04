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

#pragma once

#include <string>
#include "glog/logging.h"
#include "paddle/fluid/operators/jit/gen/jitcode.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

class SeqPoolJitCode : public JitCode {
 public:
  explicit SeqPoolJitCode(const seq_pool_attr_t& attr,
                          size_t code_size = 256 * 1024,
                          void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), h_(attr.h), w_(attr.w), type_(attr.type) {
    if (type_ != SeqPoolType::kSum) {
      LOG(FATAL) << "Only support sum pool yet ";
    }
    this->genCode();
  }

  virtual const char* name() const {
    std::string base = "SeqPoolJitCode";
    if (type_ == SeqPoolType::kSum) {
      base += "_Sum";
    } else if (type_ == SeqPoolType::kAvg) {
      base += "_Avg";
    } else if (type_ == SeqPoolType::kSqrt) {
      base += "_Sqrt";
    }
    base += ("_W" + std::to_string(w_));
    return base.c_str();
  }
  void genCode() override;

 protected:
  template <typename JMM>
  void pool_height(int w_offset, int block, int max_num_regs) {
    int offset = w_offset;
    for (int i = 0; i < max_num_regs; ++i) {
      vmovups(JMM(i), ptr[param1 + offset]);
      offset += sizeof(float) * block;
    }
    if (h_ > 1) {
      Label l_next_h;
      mov(reg_h, 1);
      mov(reg_tmp, param1);
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
        inc(reg_h);
        add(reg_tmp, w_ * sizeof(float));
        cmp(reg_h, h_);
        jl(l_next_h, T_NEAR);
      }
    }
    // save right now
    if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
      vbroadcastss(JMM(max_num_regs), reg32_scalar);
    }
    offset = w_offset;
    for (int i = 0; i < max_num_regs; ++i) {
      if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
        vmulps(JMM(i), JMM(i), JMM(max_num_regs));
      }
      vmovups(ptr[param2 + offset], JMM(i));
      offset += sizeof(float) * block;
    }
  }

  void pool_height_of_rest_width(int rest, int w_offset, int max_num_regs) {
    const int rest_used_num_regs = load_rest(rest, w_offset, 0);
    const bool has_block4 = rest / 4 > 0;
    const bool has_block2 = (rest % 4) / 2 > 0;
    const bool has_block1 = (rest % 2) == 1;
    if (h_ > 1) {
      Label l_next_h;
      mov(reg_h, 1);
      mov(reg_tmp, param1);
      add(reg_tmp, w_ * sizeof(float) + w_offset);
      L(l_next_h);
      {
        // int used_regs =load_rest(rest, h * w_ * sizeof(float) + w_offset,
        // max_num_regs);
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
        PADDLE_ENFORCE_EQ(reg_idx, rest_used_num_regs,
                          "All heights should use same regs");
        for (int i = 0; i < reg_idx; ++i) {
          vaddps(xmm_t(i), xmm_t(i), xmm_t(i + max_num_regs));
        }
        inc(reg_h);
        add(reg_tmp, w_ * sizeof(float));
        cmp(reg_h, h_);
        jl(l_next_h, T_NEAR);
      }
    }
    // save right now
    if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
      vbroadcastss(xmm_t(max_num_regs - 1), reg32_scalar);
      for (int i = 0; i < rest_used_num_regs; ++i) {
        vmulps(xmm_t(i), xmm_t(i), xmm_t(max_num_regs - 1));
      }
    }
    save_rest(rest, w_offset);
  }

  // return the number of used regs, use start from reg 0
  int load_rest(int rest, int w_offset, const int num_shift_regs,
                const int reg_start = 0) {
    const bool has_block4 = rest / 4 > 0;
    const bool has_block2 = (rest % 4) / 2 > 0;
    const bool has_block1 = (rest % 2) == 1;
    int reg_idx = reg_start;
    if (has_block4) {
      vmovups(xmm_t(reg_idx + num_shift_regs), ptr[param1 + w_offset]);
      w_offset += sizeof(float) * 4;
      reg_idx++;
    }
    if (has_block2) {
      vmovq(xmm_t(reg_idx + num_shift_regs), ptr[param1 + w_offset]);
      w_offset += sizeof(float) * 2;
      reg_idx++;
    }
    if (has_block1) {
      vmovss(xmm_t(reg_idx + num_shift_regs), ptr[param1 + w_offset]);
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
      vmovups(ptr[param2 + w_offset], xmm_t(reg_idx));
      w_offset += sizeof(float) * 4;
      reg_idx++;
    }
    if (has_block2) {
      vmovq(ptr[param2 + w_offset], xmm_t(reg_idx));
      w_offset += sizeof(float) * 2;
      reg_idx++;
    }
    if (has_block1) {
      vmovss(ptr[param2 + w_offset], xmm_t(reg_idx));
    }
  }

 private:
  int h_;
  int w_;
  SeqPoolType type_;
  reg64_t param1{abi_param1};
  reg64_t param2{abi_param2};
  reg64_t param3{abi_param3};
  reg32_t reg32_scalar{r8d};

  reg64_t reg_h{r9};
  reg64_t reg_ptr_src_i{r10};
  reg64_t reg_tmp{r11};
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
