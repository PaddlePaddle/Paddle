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

#pragma once

#include <string>
#include "paddle/fluid/operators/math/jit_gen.h"
#include "paddle/fluid/operators/math/jit_kernel_impl.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace gen {

using reg64_t = const Xbyak::Reg64;
using reg32_t = const Xbyak::Reg32;
using xmm_t = const Xbyak::Xmm;
using ymm_t = const Xbyak::Ymm;
using zmm_t = const Xbyak::Zmm;
using Label = Xbyak::Label;

typedef enum {
  mul = 0,
  add,
  sub,
  relu,
  exp,
  sigmoid,
  tanh,
  identity
} operand_type;

extern const float exp_float_consts[];
extern const int exp_int_0x7f[];
extern int g_tmp_mem[];

#define ALIGN32 __attribute__((aligned(32)))
#define EXP_HIG 88.3762626647949f
#define EXP_LOW -88.3762626647949f
#define CEPHES_LOG2EF 1.44269504088896341
#define CEPHES_EXP_C1 0.693359375
#define CEPHES_EXP_C2 -2.12194440e-4
#define CEPHES_EXP_P0 1.9875691500E-4
#define CEPHES_EXP_P1 1.3981999507E-3
#define CEPHES_EXP_P2 8.3334519073E-3
#define CEPHES_EXP_P3 4.1665795894E-2
#define CEPHES_EXP_P4 1.6666665459E-1
#define CEPHES_EXP_P5 5.0000001201E-1

#define REPEAT_8TIMES(val) val, val, val, val, val, val, val, val

#define OFFSET_EXP_ONE 0 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_TWO 1 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_0P5 2 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_HIG 3 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_LOW 4 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_LOG2EF 5 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_C1 6 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_C2 7 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P0 8 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P1 9 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P2 10 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P3 11 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P4 12 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P5 13 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_MAX_INPUT 14 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_SIGMOID_MAX 15 * YMM_FLOAT_BLOCK * sizeof(float)
#define OFFSET_SIGMOID_MIN 16 * YMM_FLOAT_BLOCK * sizeof(float)

// function: vec = Operand(vec(or scalar), vec(or scalar)) (maybe with relu)
class VXXJitCode : public JitCode {
 public:
  const char* name() const override {
    std::string base = "VXXJitCode";
    if (scalar_index_ == 1) {
      base += "_Scalar";
    } else {
      base += "_Vec";
    }
    if (type_ == operand_type::mul) {
      base += "_Mul";
    } else if (type_ == operand_type::add) {
      base += "_Add";
    }
    if (scalar_index_ == 2) {
      base += "_Scalar";
    } else {
      base += "_Vec";
    }
    base += (with_relu_ ? "_Relu" : "");
    return base.c_str();
  }
  explicit VXXJitCode(int d, operand_type type, int scalar_index,
                      bool with_relu, size_t code_size = 256 * 1024,
                      void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr),
        num_(d),
        type_(type),
        scalar_index_(scalar_index),
        with_relu_(with_relu) {}
  static bool init(int d, int scalar_index = 0);
  void generate() override;

 private:
  int num_;
  operand_type type_;
  int scalar_index_;
  bool with_relu_;
  reg64_t param1{abi_param1};
  reg64_t param2{abi_param2};
  reg64_t param3{abi_param3};

  xmm_t xmm_src1 = xmm_t(0);
  xmm_t xmm_src2 = xmm_t(1);
  xmm_t xmm_dst = xmm_t(2);
  xmm_t xmm_zero = xmm_t(3);

  ymm_t ymm_src1 = ymm_t(0);
  ymm_t ymm_src2 = ymm_t(1);
  ymm_t ymm_dst = ymm_t(2);
  ymm_t ymm_zero = ymm_t(3);
};

class VActJitCode : public JitCode {
 public:
  const char* name() const override {
    std::string base = "VActJitCode";
    switch (type_) {
      case operand_type::relu:
        base += "_Relu";
        break;
      case operand_type::exp:
        base += "_Exp";
        break;
      case operand_type::sigmoid:
        base += "_Sigmoid";
        break;
      case operand_type::tanh:
        base += "_Tanh";
        break;
      case operand_type::identity:
        base += "_Identity";
        break;
      default:
        break;
    }
    return base.c_str();
  }

  explicit VActJitCode(int d, operand_type type, size_t code_size = 256 * 1024,
                       void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), num_(d), type_(type) {}
  static bool init(int d, operand_type type);
  void generate() override;

 protected:
  // compute relu with ymm, xmm
  template <typename JMM>
  void relu_jmm(JMM& dst, JMM& src, int zero_idx = 15) {  // NOLINT
    JMM zero = JMM(zero_idx);
    vxorps(zero, zero, zero);
    vmaxps(dst, src, zero);
  }

  // compute exp with ymm, xmm
  template <typename JMM>
  void exp_jmm(JMM& dst, JMM& src, int src_idx = 11, int fx_idx = 12,  // NOLINT
               int fy_idx = 13, int mask_idx = 14, int tmp_idx = 15) {
    using namespace platform::jit;  // NOLINT
    // check all idx can not equal
    JMM jmm_src = JMM(src_idx);
    JMM jmm_fx = JMM(fx_idx);
    JMM jmm_fy = JMM(fy_idx);
    JMM jmm_mask = JMM(mask_idx);
    JMM jmm_tmp = JMM(tmp_idx);
    reg64_t reg_ptr_global = rax;
    push(reg_ptr_global);
    vmovaps(jmm_src, src);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_HIG]);
    vminps(jmm_src, jmm_src, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_LOW]);
    vmaxps(jmm_src, jmm_src, jmm_tmp);
    // express exp(x) as exp(g + n*log(2))
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_LOG2EF]);
    vmulps(jmm_fx, jmm_src, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_0P5]);
    vaddps(jmm_fx, jmm_fx, jmm_tmp);
    vroundps(jmm_fy, jmm_fx, 0x01);
    // if greater, substract 1
    vcmpgtps(jmm_mask, jmm_fy, jmm_fx);
    vmovaps(jmm_tmp, ptr[reg_ptr_global]);
    vandps(jmm_mask, jmm_mask, jmm_tmp);
    vsubps(jmm_fx, jmm_fy, jmm_mask);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_C1]);
    vmulps(jmm_fy, jmm_fx, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_C2]);
    JMM ymm_z = JMM(jmm_mask.getIdx());
    vmulps(ymm_z, jmm_fx, jmm_tmp);
    vsubps(jmm_src, jmm_src, jmm_fy);
    vsubps(jmm_src, jmm_src, ymm_z);
    vmulps(ymm_z, jmm_src, jmm_src);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_P0]);
    vmulps(dst, jmm_src, jmm_tmp);
    for (size_t i = OFFSET_EXP_P1; i < OFFSET_EXP_P5;
         i += (YMM_FLOAT_BLOCK * sizeof(float))) {
      vmovaps(jmm_tmp, ptr[reg_ptr_global + i]);  // P1~P4
      vaddps(dst, dst, jmm_tmp);
      vmulps(dst, dst, jmm_src);
    }
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_P5]);
    vaddps(dst, dst, jmm_tmp);
    vmulps(dst, dst, ymm_z);
    vaddps(dst, dst, jmm_src);
    vmovaps(jmm_tmp, ptr[reg_ptr_global]);
    vaddps(dst, dst, jmm_tmp);
    // build 2^n
    JMM ymm_int = jmm_fx;
    vcvttps2dq(ymm_int, jmm_fx);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_int_0x7f));
    vmovdqa(jmm_tmp, ptr[reg_ptr_global]);
    if (MayIUse(avx2) || std::is_same<JMM, xmm_t>::value) {
      vpaddd(ymm_int, ymm_int, jmm_tmp);
      vpslld(ymm_int, ymm_int, 23);
    } else if (MayIUse(avx)) {
      xmm_t xtmp1 = xmm_t(ymm_int.getIdx());
      xmm_t xtmp2 = xmm_t(jmm_tmp.getIdx());
      reg64_t reg_ptr_tmp = reg_ptr_global;
      mov(reg_ptr_tmp, reinterpret_cast<size_t>(g_tmp_mem));
      vmovdqa(ptr[reg_ptr_tmp], ymm_int);
      vmovdqa(ptr[reg_ptr_tmp + YMM_FLOAT_BLOCK * sizeof(float)], jmm_tmp);
      vpaddd(xtmp1, xtmp1, xtmp2);
      vpslld(xtmp1, xtmp1, 23);
      vmovdqa(ptr[reg_ptr_tmp], xtmp1);
      // next 128bits
      vmovdqa(xtmp1, ptr[reg_ptr_tmp + XMM_FLOAT_BLOCK * sizeof(float)]);
      vmovdqa(xtmp2, ptr[reg_ptr_tmp +
                         (YMM_FLOAT_BLOCK + XMM_FLOAT_BLOCK) * sizeof(float)]);
      vpaddd(xtmp1, xtmp1, xtmp2);
      vpslld(xtmp1, xtmp1, 23);
      vmovdqa(ptr[reg_ptr_tmp + XMM_FLOAT_BLOCK * sizeof(float)], xtmp1);
      // load out
      vmovdqa(ymm_int, ptr[reg_ptr_tmp]);
    }
    vmulps(dst, dst, ymm_int);
    pop(reg_ptr_global);
  }

  // compute sigmoid with ymm, xmm
  template <typename JMM>
  void sigmoid_jmm(JMM& dst, JMM& src, int src_idx = 11,  // NOLINT
                   int fx_idx = 12, int fy_idx = 13, int mask_idx = 14,
                   int tmp_idx = 15) {
    // y = 1 / (1 + e^-x)
    JMM jmm_tmp = JMM(tmp_idx);
    JMM jmm_src = JMM(src_idx);
    reg64_t reg_ptr_global = rax;
    push(reg_ptr_global);
    vmovaps(jmm_src, src);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_SIGMOID_MAX]);
    vminps(jmm_src, jmm_src, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_SIGMOID_MIN]);
    vmaxps(jmm_src, jmm_src, jmm_tmp);
    vxorps(jmm_tmp, jmm_tmp, jmm_tmp);
    vsubps(jmm_src, jmm_tmp, jmm_src);
    exp_jmm<JMM>(dst, jmm_src, src_idx, fx_idx, fy_idx, mask_idx, tmp_idx);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
    vaddps(dst, dst, jmm_tmp);
    vdivps(dst, jmm_tmp, dst);
    pop(reg_ptr_global);
  }

  // compute tanh with ymm, xmm
  template <typename JMM>
  void tanh_jmm(JMM& dst, JMM& src, int src_idx = 11,  // NOLINT
                int fx_idx = 12, int fy_idx = 13, int mask_idx = 14,
                int tmp_idx = 15) {
    // y = 2 / (1 + e^(-2x)) - 1
    JMM jmm_src = JMM(src_idx);
    JMM jmm_tmp = JMM(tmp_idx);
    JMM jmm_zero = JMM(mask_idx);
    reg64_t reg_ptr_global = rax;
    push(reg_ptr_global);
    vmovaps(jmm_src, src);
    mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_TWO]);
    vxorps(jmm_zero, jmm_zero, jmm_zero);
    vsubps(jmm_tmp, jmm_zero, jmm_tmp);
    vmulps(jmm_src, jmm_src, jmm_tmp);
    exp_jmm<JMM>(dst, jmm_src, src_idx, fx_idx, fy_idx, mask_idx, tmp_idx);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
    vaddps(dst, dst, jmm_tmp);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_TWO]);
    vdivps(dst, jmm_tmp, dst);
    vmovaps(jmm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
    vsubps(dst, dst, jmm_tmp);
    pop(reg_ptr_global);
  }

  template <typename JMM>
  void act(JMM& dst, JMM& src, operand_type type) {  // NOLINT
    // use 11~15
    switch (type) {
      case operand_type::relu:
        relu_jmm<JMM>(dst, src, 15);
        break;
      case operand_type::exp:
        exp_jmm<JMM>(dst, src, 11, 12, 13, 14, 15);
        break;
      case operand_type::sigmoid:
        sigmoid_jmm<JMM>(dst, src, 11, 12, 13, 14, 15);
        break;
      case operand_type::tanh:
        tanh_jmm<JMM>(dst, src, 11, 12, 13, 14, 15);
        break;
      case operand_type::identity:
        break;
      default:
        // throw error
        break;
    }
  }

 protected:
  int num_;
  operand_type type_;
  reg64_t param1{abi_param1};
  reg64_t param2{abi_param2};

  xmm_t xmm_src = xmm_t(0);
  ymm_t ymm_src = ymm_t(0);

  xmm_t xmm_dst = xmm_t(1);
  ymm_t ymm_dst = ymm_t(1);
};

class LSTMJitCode : public VActJitCode {
 public:
  const char* name() const override {
    std::string base = "LSTMJitCode";
    if (use_peephole_) {
      base += "_Peephole";
    }
    if (compute_c1h1_) {
      base += "_C1H1";
    }
    auto AddTypeStr = [&](operand_type type) {
      switch (type) {
        case operand_type::relu:
          base += "_Relu";
          break;
        case operand_type::exp:
          base += "_Exp";
          break;
        case operand_type::sigmoid:
          base += "_Sigmoid";
          break;
        case operand_type::tanh:
          base += "_Tanh";
          break;
        case operand_type::identity:
          base += "_Identity";
          break;
        default:
          break;
      }
    };
    AddTypeStr(act_gate_);
    AddTypeStr(act_cand_);
    AddTypeStr(act_cell_);
    return base.c_str();
  }

  explicit LSTMJitCode(bool compute_c1h1, const lstm_attr_t& attr,
                       size_t code_size = 256 * 1024, void* code_ptr = nullptr)
      : VActJitCode(attr.d, operand_type::sigmoid /* this is bugy*/, code_size,
                    code_ptr),
        compute_c1h1_(compute_c1h1) {
    auto typeExchange = [](const std::string& type) -> gen::operand_type {
      if (type == "sigmoid") {
        return operand_type::sigmoid;
      } else if (type == "relu") {
        return operand_type::relu;
      } else if (type == "tanh") {
        return operand_type::tanh;
      } else if (type == "identity" || type == "") {
        return operand_type::identity;
      }  // else throw error
      return operand_type::identity;
    };
    num_ = attr.d;
    use_peephole_ = attr.use_peephole;
    act_gate_ = typeExchange(attr.act_gate);
    act_cand_ = typeExchange(attr.act_cand);
    act_cell_ = typeExchange(attr.act_cell);
  }
  static bool init(int d);
  void generate() override;

 protected:
  int num_;
  bool compute_c1h1_;
  bool use_peephole_;
  operand_type act_gate_;
  operand_type act_cand_;
  operand_type act_cell_;
  reg64_t param1{abi_param1};
};

class GRUJitCode : public VActJitCode {
 public:
  const char* name() const override {
    std::string base = "GRUJitCode";
    if (id_ == 0) {
      base += "_H1";
    } else if (id_ == 1) {
      base += "_HtPart1";
    } else if (id_ == 2) {
      base += "_HtPart2";
    }
    auto AddTypeStr = [&](operand_type type) {
      switch (type) {
        case operand_type::relu:
          base += "_Relu";
          break;
        case operand_type::exp:
          base += "_Exp";
          break;
        case operand_type::sigmoid:
          base += "_Sigmoid";
          break;
        case operand_type::tanh:
          base += "_Tanh";
          break;
        case operand_type::identity:
          base += "_Identity";
          break;
        default:
          break;
      }
    };
    AddTypeStr(act_gate_);
    AddTypeStr(act_cand_);
    return base.c_str();
  }

  explicit GRUJitCode(int id, const gru_attr_t& attr,
                      size_t code_size = 256 * 1024, void* code_ptr = nullptr)
      : VActJitCode(attr.d, operand_type::sigmoid /* this is bugy*/, code_size,
                    code_ptr),
        id_(id) {
    auto typeExchange = [](const std::string& type) -> gen::operand_type {
      if (type == "sigmoid") {
        return operand_type::sigmoid;
      } else if (type == "relu") {
        return operand_type::relu;
      } else if (type == "tanh") {
        return operand_type::tanh;
      } else if (type == "identity" || type == "") {
        return operand_type::identity;
      }  // else throw error
      return operand_type::identity;
    };
    num_ = attr.d;
    act_gate_ = typeExchange(attr.act_gate);
    act_cand_ = typeExchange(attr.act_cand);
  }
  static bool init(int d);
  void generate() override;

 protected:
  int id_;
  int num_;
  operand_type act_gate_;
  operand_type act_cand_;
  reg64_t param1{abi_param1};
};

#ifdef PADDLE_WITH_MKLDNN
struct EltwiseMulnChw16cNC : public Xbyak::CodeGenerator {
  explicit EltwiseMulnChw16cNC(size_t code_size = 256 * 1024)
      : Xbyak::CodeGenerator(code_size) {
    // RDI is ptr x_input
    // RSI is ptr y_input
    // RDX is ptr output
    // RCX is height
    // r8 is width

    push(rbx);

    xor_(rax, rax);
    xor_(r10, r10);
    vmovups(zmm3, ptr[rsi]);

    L("h_loop");
    xor_(rbx, rbx);
    L("w_loop");
    vmovups(zmm2, ptr[rdi + rax]);
    vmulps(zmm1, zmm2, zmm3);
    vmovups(ptr[rdx + rax], zmm1);
    add(rax, 64);
    inc(rbx);
    cmp(r8, rbx);
    jnz("w_loop");
    inc(r10);
    cmp(r10, rcx);
    jnz("h_loop");

    pop(rbx);
    ret();
  }
};
#endif

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
