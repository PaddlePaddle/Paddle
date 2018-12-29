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
    // TODO(TJ): make h load from params
    base += ("_H" + std::to_string(h_));
    return base.c_str();
  }
  void genCode() override;

 protected:
  template <typename JMM>
  void pool_height(int w_offset, int block, int max_num_regs) {
    for (int h = 0; h < h_; ++h) {
      int offset = h * w_ * sizeof(float) + w_offset;
      const int shift_regs = (h == 0) ? 0 : max_num_regs;
      for (int i = 0; i < max_num_regs; ++i) {
        vmovups(JMM(i + shift_regs), ptr[param1 + offset]);
        offset += sizeof(float) * block;
      }
      if (h > 0) {
        // sum anyway
        for (int i = 0; i < max_num_regs; ++i) {
          vaddps(JMM(i), JMM(i), JMM(i + max_num_regs));
        }
      }
    }
    // save right now
    if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
      vbroadcastss(JMM(max_num_regs), reg32_scalar);
    }
    int offset = w_offset;
    for (int i = 0; i < max_num_regs; ++i) {
      if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
        vmulps(JMM(i), JMM(i), JMM(max_num_regs));
      }
      vmovups(ptr[param2 + offset], JMM(i));
      offset += sizeof(float) * block;
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
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
