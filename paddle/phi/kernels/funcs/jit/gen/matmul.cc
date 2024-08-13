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

#include "paddle/phi/kernels/funcs/jit/gen/matmul.h"

#include <cstddef>  // offsetof

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::gen {

void MatMulJitCode::genCode() {
  preCode();
  int block, rest;
  const auto groups = packed_groups(n_, k_, &block, &rest);
  PADDLE_ENFORCE_GT(
      groups.front(),
      0,
      common::errors::InvalidArgument("The number of rest registers should "
                                      "be larger than 0. But it is %d.",
                                      groups.front()));

  const int block_len = sizeof(float) * block;  // NOLINT
  const int x_reg_idx = (block == ZMM_FLOAT_BLOCK ? 32 : 16) - 1;
  const int w_reg_idx = x_reg_idx - 1;
  // from packed mov(reg_ptr_wgt, ptr[param_attr + offsetof(matmul_attr_t,
  // packed_weight)]);
  mov(reg_ptr_wgt, param_y);
  size_t z_offset = 0;
  size_t wgt_offset = 0;
  for (size_t g = 0; g < groups.size(); ++g) {
    size_t x_offset = 0;
    size_t wgt_offset_tmp = 0;
    for (size_t i = 0; i < g; ++i) {
      wgt_offset_tmp += groups[i] * block_len;
    }
    for (int k = 0; k < k_; ++k) {
      wgt_offset = wgt_offset_tmp;
      vbroadcastss(zmm_t(x_reg_idx), ptr[param_x + x_offset]);
      // clean
      if (k == 0) {
        for (int i = 0; i < groups[g]; ++i) {
          vxorps(zmm_t(i), zmm_t(i), zmm_t(i));
        }
      }
      for (int i = 0; i < groups[g]; ++i) {
        vmovups(zmm_t(w_reg_idx),
                ptr[reg_ptr_wgt + wgt_offset + k * n_ * sizeof(float)]);
        vfmadd231ps(zmm_t(i), zmm_t(w_reg_idx), zmm_t(x_reg_idx));
        wgt_offset += block_len;
      }
      // last one, save
      if (k == k_ - 1) {
        for (int i = 0; i < groups[g]; ++i) {
          // only rest save should be careful
          if (rest != 0 && g == groups.size() - 1 && i == groups[g] - 1) {
            break;
          }
          vmovups(ptr[param_z + z_offset + i * block_len], zmm_t(i));
        }
      }
      x_offset += sizeof(float);
    }
    z_offset += block_len * groups[g];
  }

  if (rest != 0) {
    // below should refine with mask
    int reg_idx = groups.back() - 1;
    z_offset = (n_ - rest) * sizeof(float);
    int inner_block = 8;
    while (rest > 0) {
      if (rest >= 8) {
        inner_block = 8;
        vmovups(ptr[param_z + z_offset], ymm_t(reg_idx));
        // shift zmm of inner_block, change reg_idx if update
      } else if (rest >= 4) {
        inner_block = 4;
        vmovups(ptr[param_z + z_offset], xmm_t(reg_idx));
      } else if (rest >= 2) {
        inner_block = 2;
        vmovq(ptr[param_z + z_offset], xmm_t(reg_idx));
      } else {
        inner_block = 1;
        vmovss(ptr[param_z + z_offset], xmm_t(reg_idx));
      }
      z_offset += inner_block * sizeof(float);
      rest -= inner_block;
    }
  }

  postCode();
}

class MatMulCreator : public JitCodeCreator<matmul_attr_t> {
 public:
  bool CanBeUsed(const matmul_attr_t& attr) const override {
    return attr.m == 1 &&
           phi::backends::cpu::MayIUse(phi::backends::cpu::avx512f) &&
           attr.n % ZMM_FLOAT_BLOCK == 0 && attr.k < 512;
  }
  size_t CodeSize(const matmul_attr_t& attr) const override {
    int block = YMM_FLOAT_BLOCK;
    if (phi::backends::cpu::MayIUse(phi::backends::cpu::avx512f)) {
      block = ZMM_FLOAT_BLOCK;
    }
    return 96 + 4 * attr.k * (attr.n / block + 1) * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const matmul_attr_t& attr) const override {
    PADDLE_ENFORCE_GT(
        attr.m,
        0,
        common::errors::InvalidArgument(
            "The attribute m (first matrix's row) of MatMul should "
            "be larger than 0. But it is %d.",
            attr.m));
    PADDLE_ENFORCE_GT(
        attr.n,
        0,
        common::errors::InvalidArgument(
            "The attribute n (first matrix's col) of MatMul should "
            "be larger than 0. But it is %d.",
            attr.n));
    PADDLE_ENFORCE_GT(
        attr.k,
        0,
        common::errors::InvalidArgument(
            "The attribute k (second matrix's col) of MatMul should "
            "be larger than 0. But it is %d.",
            attr.k));
    return make_unique<MatMulJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace phi::jit::gen

namespace gen = phi::jit::gen;

REGISTER_JITKERNEL_GEN(kMatMul, gen::MatMulCreator);
