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

#include "paddle/fluid/operators/jit/gen/embseqpool.h"

#include <stddef.h>  // offsetof

#include "paddle/fluid/operators/jit/macro.h"
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void EmbSeqPoolJitCode::genCode() {
  preCode();
  constexpr int block = YMM_FLOAT_BLOCK;
  constexpr int max_num_regs = 8;
  const int num_block = tbl_w_ / block;
  const int num_groups = num_block / max_num_regs;
  const size_t block_size = sizeof(float) * block;
  std::vector<int> groups(num_groups, max_num_regs);
  int rest_num_regs = num_block % max_num_regs;
  if (rest_num_regs > 0) {
    groups.push_back(rest_num_regs);
  }

  // protect param_dst
  mov(reg_ptr_param_dst, param_dst);
  mov(reg_idx_width_in_byte,
      qword[param_attr + offsetof(emb_seq_pool_attr_t, index_width)]);
  mov(reg_idx_height,
      qword[param_attr + offsetof(emb_seq_pool_attr_t, index_height)]);
  mov(rax, sizeof(int64_t));
  mul(reg_idx_width_in_byte);
  mov(reg_idx_width_in_byte, rax);
  const size_t tbl_width_in_byte = sizeof(float) * tbl_w_;
  int acc_num_regs = 0;
  for (int num_regs : groups) {
    Label l_next_idx_w, l_next_idx_h, l_save_now;
    xor_(reg_idx_w_i_in_byte, reg_idx_w_i_in_byte);
    mov(reg_ptr_dst_i, reg_ptr_param_dst);
    add(reg_ptr_dst_i, acc_num_regs * block_size);

    L(l_next_idx_w);
    {
      // h == 0
      mov(reg_ptr_idx_i, param_idx);
      add(reg_ptr_idx_i, reg_idx_w_i_in_byte);
      mov(reg_idx, qword[reg_ptr_idx_i]);
      mov(rax, tbl_width_in_byte);
      mul(reg_idx);
      mov(reg_ptr_tbl_i, rax);        // reg is offset now
      add(reg_ptr_tbl_i, param_tbl);  // reg is ptr_i now
      size_t w_offset = 0;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ymm_t(reg_i + num_regs), ptr[reg_ptr_tbl_i + w_offset]);
        w_offset += block_size;
      }
      add(reg_ptr_idx_i, reg_idx_width_in_byte);

      // end condition of idx h
      mov(reg_idx_h_end, reg_idx_height);
      mov(rax, reg_idx_width_in_byte);
      mul(reg_idx_h_end);
      mov(reg_idx_h_end, rax);
      add(reg_idx_h_end, reg_idx_w_i_in_byte);
      add(reg_idx_h_end, param_idx);

      cmp(reg_ptr_idx_i, reg_idx_h_end);
      jge(l_save_now, T_NEAR);
      L(l_next_idx_h);
      {
        mov(reg_idx, qword[reg_ptr_idx_i]);
        mov(reg_ptr_tbl_i, reg_idx);
        mov(rax, tbl_width_in_byte);
        mul(reg_idx);
        mov(reg_ptr_tbl_i, rax);
        add(reg_ptr_tbl_i, param_tbl);
        size_t w_offset = 0;
        for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
          vmovups(ymm_t(reg_i), ptr[reg_ptr_tbl_i + w_offset]);
          vaddps(
              ymm_t(reg_i + num_regs), ymm_t(reg_i + num_regs), ymm_t(reg_i));
          w_offset += block_size;
        }
        add(reg_ptr_idx_i, reg_idx_width_in_byte);
        cmp(reg_ptr_idx_i, reg_idx_h_end);
        jl(l_next_idx_h, T_NEAR);
      }  // end of idx h
      L(l_save_now);
      // avg or sqrt here, if needed
      w_offset = 0;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ptr[reg_ptr_dst_i + w_offset], ymm_t(reg_i + num_regs));
        w_offset += block_size;
      }
      add(reg_ptr_dst_i, tbl_width_in_byte);
      add(reg_idx_w_i_in_byte, sizeof(int64_t));
      cmp(reg_idx_w_i_in_byte, reg_idx_width_in_byte);
      jl(l_next_idx_w, T_NEAR);
    }  // end of idx w

    acc_num_regs += num_regs;
    add(param_tbl, num_regs * block_size);  // do not use acc_num_regs
  }                                         // end of groups
  postCode();
}

class EmbSeqPoolCreator : public JitCodeCreator<emb_seq_pool_attr_t> {
 public:
  bool CanBeUsed(const emb_seq_pool_attr_t& attr) const override {
    return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) &&
           attr.table_width % YMM_FLOAT_BLOCK == 0;
  }
  size_t CodeSize(const emb_seq_pool_attr_t& attr) const override {
    return 96 + (attr.table_width / YMM_FLOAT_BLOCK) * 96 * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const emb_seq_pool_attr_t& attr) const override {
    PADDLE_ENFORCE_GT(attr.table_height,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute table_height of EmbSeqPool should "
                          "be larger than 0. But it is %d.",
                          attr.table_height));
    PADDLE_ENFORCE_GT(attr.table_width,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute table_width of EmbSeqPool should "
                          "be larger than 0. But it is %d.",
                          attr.table_width));
    PADDLE_ENFORCE_GT(attr.index_height,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute index_height of EmbSeqPool should "
                          "be larger than 0. But it is %d.",
                          attr.index_height));
    PADDLE_ENFORCE_GT(attr.index_width,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute index_width of EmbSeqPool should "
                          "be larger than 0. But it is %d.",
                          attr.index_width));
    PADDLE_ENFORCE_GT(attr.out_width,
                      0,
                      platform::errors::InvalidArgument(
                          "The attribute out_width of EmbSeqPool should be "
                          "larger than 0. But it is %d.",
                          attr.out_width));
    return make_unique<EmbSeqPoolJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kEmbSeqPool, gen::EmbSeqPoolCreator);
