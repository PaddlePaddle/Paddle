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

class EmbSeqPoolJitCode : public JitCode {
 public:
  explicit EmbSeqPoolJitCode(const emb_seq_pool_attr_t& attr,
                             size_t code_size = 256 * 1024,
                             void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr),
        tbl_w_(attr.table_width),
        type_(attr.pool_type) {
    if (type_ != SeqPoolType::kSum) {
      PADDLE_THROW(
          common::errors::Unimplemented("Only supports sum pool yet."));
    }
    this->genCode();
  }

  std::string name() const override {
    std::string base = "EmbSeqPoolJitCode";
    if (type_ == SeqPoolType::kSum) {
      base += "_Sum";
    } else if (type_ == SeqPoolType::kAvg) {
      base += "_Avg";
    } else if (type_ == SeqPoolType::kSqrt) {
      base += "_Sqrt";
    }
    base += ("_W" + std::to_string(tbl_w_));
    return base;
  }
  void genCode() override;

 private:
  int tbl_w_;
  SeqPoolType type_;
  reg64_t param_tbl{abi_param1};
  reg64_t param_idx{abi_param2};
  reg64_t param_dst{abi_param3};
  reg64_t param_attr{abi_param4};

  reg64_t reg_tmp{rax};

  reg64_t reg_idx_width_in_byte{r8};
  reg64_t reg_idx_height{r9};

  reg64_t reg_ptr_tbl_i{r10};
  reg64_t reg_idx{r10};  // could use same of reg_ptr_tbl_i
  reg64_t reg_ptr_idx_i{r11};
  reg64_t reg_ptr_dst_i{r12};
  reg64_t reg_ptr_param_dst{r13};  // rdx is used in mul so protect param_dst

  reg64_t reg_idx_w_i_in_byte{r14};
  reg64_t reg_idx_h_end{r15};
};

}  // namespace gen
}  // namespace jit
}  // namespace phi
