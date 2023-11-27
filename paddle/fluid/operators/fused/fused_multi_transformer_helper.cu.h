/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/fused/attn_gemm_int8.h"
// #include
// "paddle/fluid/operators/fused/cutlass/cutlass_kernels/gemm_dequant.h"
// #include
// "paddle/fluid/operators/fused/cutlass/cutlass_kernels/intA_intB_interleaved_gemm/intA_intB_gemm_template.h"
#include "paddle/fluid/operators/fused/fused_multi_transformer_op.cu.h"

DECLARE_int64(custom_allreduce_one_shot_threshold);
DECLARE_int64(custom_allreduce_two_shot_threshold);

DECLARE_bool(use_gemm_dequant);

DECLARE_bool(print_matrix);
/*
Note(Zhengzekang):
This header file is to store General Function Helper which has been used in
FusedMultiTransformer.
*/

namespace paddle {
namespace operators {

template <typename T>
static void PrintFrontNPerLine(const phi::DenseTensor &a,
                               int rows,
                               int cols,
                               int n) {
  if (!FLAGS_print_matrix) return;
  std::vector<T> a_h(a.numel());

  cudaMemcpy(
      a_h.data(), a.data<T>(), a.numel() * sizeof(T), cudaMemcpyDeviceToHost);

  for (int line = 0; line < rows; ++line) {
    std::cout << "[" << line << "] ";
    for (int i = 0; i < n; ++i) {
      if (std::is_same<T, int8_t>::value) {
        std::cout << (int)(a_h[line * cols + i]) << " ";  // NOLINT
      } else {
        std::cout << a_h[line * cols + i] << " ";  // NOLINT
      }
    }
    std::cout << "\n";
  }
}

static CustomNCCLComm *GetCustomNCCLComm(const phi::GPUContext &ctx,
                                         int ring_id) {
  static auto comm =
      CreateCustomNCCLComm(ctx,
                           FLAGS_custom_allreduce_one_shot_threshold,
                           FLAGS_custom_allreduce_two_shot_threshold,
                           ring_id);
  return comm.get();
}
template <typename T>
struct AddTriFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b, const T c) const {
    return a + b + c;
  }
};

template <typename T>
struct SmoothFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b, const T c) const {
    return (a + b) * c;
  }
};

namespace {  // NOLINT

template <typename T>
class BiasActHelper {
 public:
  BiasActHelper(const phi::GPUContext &dev_ctx,
                const std::string &act_method,
                int rows,
                int cols)
      : dev_ctx_(dev_ctx), act_method_(act_method), rows_(rows), cols_(cols) {}

  // dst = Activation(x + bias(optional))
  void Compute(const phi::DenseTensor *x,
               const phi::DenseTensor *bias,
               phi::DenseTensor *output) {
    const T *bias_data = (bias == nullptr) ? nullptr : bias->data<T>();
    Load<T> load_func(x->data<T>());
    Store<T> store_func(output->data<T>());
    ComputeImpl(bias_data, load_func, store_func);
  }

  void Compute(const phi::DenseTensor *x,
               const phi::DenseTensor *bias,
               const phi::DenseTensor *dequant_scales,
               const phi::DenseTensor *shift,
               const phi::DenseTensor *smooth,
               const float quant_scale,
               const int quant_round_type,
               const float quant_max_bound,
               const float quant_min_bound,
               phi::DenseTensor *output) {
    if (shift != nullptr) {
      DispatchComputeImpl(x,
                          bias,
                          dequant_scales,
                          shift,
                          smooth,
                          quant_scale,
                          quant_round_type,
                          quant_max_bound,
                          quant_min_bound,
                          output);
    } else {
      DispatchComputeImpl(x,
                          bias,
                          dequant_scales,
                          quant_scale,
                          quant_round_type,
                          quant_max_bound,
                          quant_min_bound,
                          output);
    }
  }

 private:
  void DispatchComputeImpl(const phi::DenseTensor *x,
                           const phi::DenseTensor *bias,
                           const phi::DenseTensor *dequant_scales,
                           const float quant_scale,
                           const int quant_round_type,
                           const float quant_max_bound,
                           const float quant_min_bound,
                           phi::DenseTensor *output) {
    const T *bias_data = (bias == nullptr) ? nullptr : bias->data<T>();
    if (dequant_scales != nullptr && quant_scale > 0) {
      DequantLoad<T> load_func(
          x->data<int32_t>(), dequant_scales->data<float>(), cols_);
      QuantStore<T> store_func(output->data<int8_t>(),
                               quant_round_type,
                               quant_scale,
                               quant_max_bound,
                               quant_min_bound);
      ComputeImpl<DequantLoad<T>, QuantStore<T>, int32_t>(
          bias_data, load_func, store_func);
    } else if (dequant_scales == nullptr && quant_scale > 0) {
      Load<T> load_func(x->data<T>());
      QuantStore<T> store_func(output->data<int8_t>(),
                               quant_round_type,
                               quant_scale,
                               quant_max_bound,
                               quant_min_bound);
      ComputeImpl(bias_data, load_func, store_func);
    } else if (dequant_scales != nullptr && quant_scale <= 0) {
      DequantLoad<T> load_func(
          x->data<int32_t>(), dequant_scales->data<float>(), cols_);
      Store<T> store_func(output->data<T>());
      ComputeImpl<DequantLoad<T>, Store<T>, int32_t>(
          bias_data, load_func, store_func);
    } else {
      Load<T> load_func(x->data<T>());
      Store<T> store_func(output->data<T>());
      ComputeImpl(bias_data, load_func, store_func);
    }
  }

  void DispatchComputeImpl(const phi::DenseTensor *x,
                           const phi::DenseTensor *bias,
                           const phi::DenseTensor *dequant_scales,
                           const phi::DenseTensor *shift,
                           const phi::DenseTensor *smooth,
                           const float quant_scale,
                           const int quant_round_type,
                           const float quant_max_bound,
                           const float quant_min_bound,
                           phi::DenseTensor *output) {
    bool use_glu = (act_method_ == "geglu" || act_method_ == "swiglu");
    const T *bias_data = (bias == nullptr) ? nullptr : bias->data<T>();
    if (dequant_scales != nullptr && quant_scale > 0) {
      DequantLoad<T> load_func(
          x->data<int32_t>(), dequant_scales->data<float>(), cols_);
      QuantStore<T, true> store_func(output->data<int8_t>(),
                                     shift->data<T>(),
                                     smooth->data<T>(),
                                     use_glu ? cols_ / 2 : cols_,
                                     quant_round_type,
                                     quant_scale,
                                     quant_max_bound,
                                     quant_min_bound);
      ComputeImpl<DequantLoad<T>, QuantStore<T, true>, int32_t>(
          bias_data, load_func, store_func);
    } else if (dequant_scales == nullptr && quant_scale > 0) {
      Load<T> load_func(x->data<T>());
      QuantStore<T, true> store_func(output->data<int8_t>(),
                                     shift->data<T>(),
                                     smooth->data<T>(),
                                     use_glu ? cols_ / 2 : cols_,
                                     quant_round_type,
                                     quant_scale,
                                     quant_max_bound,
                                     quant_min_bound);
      ComputeImpl(bias_data, load_func, store_func);
    } else if (dequant_scales != nullptr && quant_scale <= 0) {
      DequantLoad<T> load_func(
          x->data<int32_t>(), dequant_scales->data<float>(), cols_);
      Store<T, true> store_func(output->data<T>(),
                                shift->data<T>(),
                                smooth->data<T>(),
                                use_glu ? cols_ / 2 : cols_);
      ComputeImpl<DequantLoad<T>, Store<T, true>, int32_t>(
          bias_data, load_func, store_func);
    } else {
      Load<T> load_func(x->data<T>());
      Store<T, true> store_func(output->data<T>(),
                                shift->data<T>(),
                                smooth->data<T>(),
                                use_glu ? cols_ / 2 : cols_);
      ComputeImpl(bias_data, load_func, store_func);
    }
  }

  template <typename LoadFunc, typename StoreFunc, typename LoadT = T>
  void ComputeImpl(const T *bias_data,
                   LoadFunc load_func,
                   StoreFunc store_func) {
    if (act_method_ == "geglu") {
      // Note(Zhengzekang): For GLU structure, we need divide the cols by 2.
      VLOG(5) << "doing geglu";
      LaunchActFFNGlu<T, GeluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
          dev_ctx_, bias_data, rows_, cols_ / 2, load_func, store_func);
    } else if (act_method_ == "swiglu") {
      VLOG(5) << "doing swiglu";
      LaunchActFFNGlu<T, CudaSwishFunctor<T>, LoadFunc, StoreFunc, LoadT>(
          dev_ctx_, bias_data, rows_, cols_ / 2, load_func, store_func);
    } else if (act_method_ == "gelu") {
      if (FLAGS_use_fast_math) {
        VLOG(5) << "doing Fast GELU";
        LaunchBiasAct<T, FastGeluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
            dev_ctx_, bias_data, rows_, cols_, load_func, store_func);
      } else {
        VLOG(5) << "doing GELU";
        LaunchBiasAct<T, GeluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
            dev_ctx_, bias_data, rows_, cols_, load_func, store_func);
      }
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently Only Support GeGLU, SwiGLU, GeLU"));
    }
  }
  const phi::GPUContext &dev_ctx_;
  std::string act_method_;
  int rows_;
  int cols_;
};

template <typename T, typename nvT = typename PDDataTypeTraits<T>::DataType>
class GEMMHelper {
 public:
  GEMMHelper(
      const phi::GPUContext &dev_ctx,
      int token_num,
      int dim_ffn,
      int dim_embed,
      const std::string gemm_method,
      // paddle::operators::CutlassFpAIntBGemmRunner<nvT, uint8_t>
      //     *int8_mixed_gemm_runner,
      // paddle::operators::CutlassFpAIntBGemmRunner<nvT, cutlass::uint4b_t>
      //     *int4_mixed_gemm_runner,
      // paddle::operators::CutlassIntAIntBInterleavedGemmRunner<nvT, int8_t>
      //     *int8_int8_interleaved_gemm_runner,
      bool transpose_weight = false)
      : dev_ctx_(dev_ctx),
        token_num_(token_num),
        dim_ffn_(dim_ffn),
        dim_embed_(dim_embed),
        gemm_method_(gemm_method),
        // int8_mixed_gemm_runner_(int8_mixed_gemm_runner),
        // int4_mixed_gemm_runner_(int4_mixed_gemm_runner),
        // int8_int8_interleaved_gemm_runner_(int8_int8_interleaved_gemm_runner),
        transpose_weight_(transpose_weight) {}

  // dst = act(fc(src[0]) + bias) * src[1]
  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,
               const phi::DenseTensor *scale,
               const phi::DenseTensor *bias,
               phi::DenseTensor *workspace,
               phi::DenseTensor *output) {
    VLOG(5) << "GEMMHelper,"
            << " token_num_:" << token_num_ << " dim_ffn_:" << dim_ffn_
            << " dim_embed_:" << dim_embed_;
    bool compute_bias = true;
    if (bias == nullptr) {
      compute_bias = false;
    }
    using NvType = typename PDDataTypeTraits<T>::DataType;

    if (gemm_method_ == "weight-only") {
      // VLOG(5) << "do weight-only gemm int8";
      // if (bias) {
      //   int8_mixed_gemm_runner_->gemm_bias_act(
      //       reinterpret_cast<const NvType *>(input->data<T>()),
      //       reinterpret_cast<const uint8_t *>(weight->data<int8_t>()),
      //       reinterpret_cast<const NvType *>(scale->data<T>()),
      //       reinterpret_cast<const NvType *>(bias->data<T>()),
      //       reinterpret_cast<NvType *>(output->data<T>()),
      //       token_num_,
      //       dim_ffn_,
      //       dim_embed_,
      //       "none",
      //       reinterpret_cast<char *>(workspace->data<uint8_t>()),
      //       workspace->numel(),
      //       dev_ctx_.stream());
      // } else {
      //   int8_mixed_gemm_runner_->gemm(
      //       reinterpret_cast<const NvType *>(input->data<T>()),
      //       reinterpret_cast<const uint8_t *>(weight->data<int8_t>()),
      //       reinterpret_cast<const NvType *>(scale->data<T>()),
      //       reinterpret_cast<NvType *>(output->data<T>()),
      //       token_num_,
      //       dim_ffn_,
      //       dim_embed_,
      //       reinterpret_cast<char *>(workspace->data<uint8_t>()),
      //       workspace->numel(),
      //       dev_ctx_.stream());
      // }
      // VLOG(5) << "input:" << *input;
      // VLOG(5) << "output:" << *output;
    } else if (gemm_method_ == "weight-only-int4") {
      // VLOG(5) << "do weight-only gemm";
      // if (bias) {
      //   int4_mixed_gemm_runner_->gemm_bias_act(
      //       reinterpret_cast<const NvType *>(input->data<T>()),
      //       reinterpret_cast<const cutlass::uint4b_t
      //       *>(weight->data<int8_t>()), reinterpret_cast<const NvType
      //       *>(scale->data<T>()), reinterpret_cast<const NvType
      //       *>(bias->data<T>()), reinterpret_cast<NvType
      //       *>(output->data<T>()), token_num_, dim_ffn_, dim_embed_, "none",
      //       reinterpret_cast<char *>(workspace->data<uint8_t>()),
      //       workspace->numel(),
      //       dev_ctx_.stream());
      // } else {
      //   int4_mixed_gemm_runner_->gemm(
      //       reinterpret_cast<const NvType *>(input->data<T>()),
      //       reinterpret_cast<const cutlass::uint4b_t
      //       *>(weight->data<int8_t>()), reinterpret_cast<const NvType
      //       *>(scale->data<T>()), reinterpret_cast<NvType
      //       *>(output->data<T>()), token_num_, dim_ffn_, dim_embed_,
      //       reinterpret_cast<char *>(workspace->data<uint8_t>()),
      //       workspace->numel(),
      //       dev_ctx_.stream());
      // }
      // VLOG(5) << "input:" << *input;
      // VLOG(5) << "output:" << *output;
    } else if (gemm_method_ == "weightonly_gemv") {
      // // TODO(zhengzekang): support weightonly gemv int4
      // const T *bias_data = bias ? bias->data<T>() : nullptr;
      // phi::GemvWeightonlyInt8Wrapper<T, phi::GPUContext>(dev_ctx_,
      //                                                    input->data<T>(),
      //                                                    weight->data<int8_t>(),
      //                                                    bias_data,
      //                                                    scale->data<T>(),
      //                                                    dim_ffn_,
      //                                                    dim_embed_,
      //                                                    "None",
      //                                                    /*act_method*/
      //                                                    output->data<T>());
    } else if (gemm_method_ == "LLM.int8") {
      // // Note(Zhengzekang): LLM Gemm donot support fused add_bias.
      // LLMGemm<T, nvT>(dev_ctx_,
      //                 weight,
      //                 input,
      //                 scale,
      //                 int8_int8_interleaved_gemm_runner_,
      //                 FLAGS_custom_llm_int8_threshold,
      //                 output,
      //                 workspace,
      //                 "LLMGemm",
      //                 token_num_,
      //                 dim_embed_,
      //                 dim_ffn_);
    } else if (gemm_method_ == "None") {
      auto ffn_linear_compute = AttnMatMul<T>(dev_ctx_,
                                              false,
                                              transpose_weight_,
                                              token_num_,
                                              dim_ffn_,
                                              dim_embed_,
                                              compute_bias);
      ffn_linear_compute.ComputeForward(weight, input, bias, output, output);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently GemmHelper only support `weight-only`, `LLM.int8`, "
          "`None`. "));
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int token_num_;
  int dim_ffn_;
  int dim_embed_;
  std::string gemm_method_;
  // paddle::operators::CutlassFpAIntBGemmRunner<nvT, uint8_t>
  //     *int8_mixed_gemm_runner_;
  // paddle::operators::CutlassFpAIntBGemmRunner<nvT, cutlass::uint4b_t>
  //     *int4_mixed_gemm_runner_;
  // paddle::operators::CutlassIntAIntBInterleavedGemmRunner<nvT, int8_t>
  //     *int8_int8_interleaved_gemm_runner_;
  bool transpose_weight_;  // Just For AttnMatmul.
};

template <typename T>
class Int8GEMMHelper {
 public:
  Int8GEMMHelper(const phi::GPUContext &dev_ctx,
                 int m,
                 int k,
                 int n,
                 phi::DenseTensor &workspace,        // NOLINT
                 phi::DenseTensor &input_workspace,  // NOLINT
                 phi::DenseTensor &out_workspace,    // NOLINT
                 int quant_round_type,
                 float quant_max_bound,
                 float quant_min_bound,
                 bool use_gemm_dequant = false)
      : dev_ctx_(dev_ctx),
        m_(m),
        k_(k),
        n_(n),
        use_gemm_dequant_(use_gemm_dequant),
        quant_round_type_(quant_round_type),
        quant_min_bound_(quant_min_bound),
        quant_max_bound_(quant_max_bound),
        workspace_(workspace),
        input_workspace_(input_workspace),
        out_workspace_(out_workspace) {
    cublaslt_helper = std::make_unique<CublasLtHelper<int32_t>>(
        m, k, n, dev_ctx.cublaslt_handle());
  }

  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,  // int8, Need be transposed
               const phi::DenseTensor *dequant_out_scales,
               const float quant_in_scale,
               phi::DenseTensor *output,
               bool quant_in = false,
               bool dequant_out = false) {
    phi::DenseTensor input_tmp, out_tmp;
    if (quant_in) {
      input_tmp = input_workspace_;
      quantize_kernel_launcher<T>(input->data<T>(),
                                  input_tmp.data<int8_t>(),
                                  quant_in_scale,
                                  m_,
                                  k_,
                                  quant_round_type_,
                                  quant_max_bound_,
                                  quant_min_bound_,
                                  dev_ctx_.stream());
    } else {
      input_tmp = *input;
    }

    if (dequant_out) {
      out_tmp = out_workspace_;
    } else {
      out_tmp = *output;
    }

    if (use_gemm_dequant_ && dequant_out) {
      // RunGemmDequant<T>(input_tmp.data<int8_t>(),
      //                   weight->data<int8_t>(),
      //                   dequant_out_scales->data<float>(),
      //                   output->data<T>(),
      //                   m_,
      //                   k_,
      //                   n_,
      //                   dev_ctx_.stream());
    } else {
      cublaslt_helper->GEMM(input_tmp.data<int8_t>(),
                            weight->data<int8_t>(),
                            out_tmp.data<int32_t>(),
                            dev_ctx_.stream(),
                            (void *)workspace_.data<int8_t>(),
                            workspace_.numel());
    }

    if (!use_gemm_dequant_ && dequant_out) {
      auto gpu_config = std::make_unique<GpuLaunchConfig>(
          phi::backends::gpu::GetGpuLaunchConfig1D(
              dev_ctx_, m_ * n_, DequantKernelVecSize));
      dequantize_kernel_launcher<T>(out_tmp.data<int32_t>(),
                                    output->data<T>(),
                                    m_,
                                    n_,
                                    dev_ctx_.stream(),
                                    gpu_config.get(),
                                    quant_in_scale,
                                    dequant_out_scales->data<float>());
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int m_;
  int k_;
  int n_;
  int quant_round_type_;
  float quant_max_bound_;
  float quant_min_bound_;
  bool use_gemm_dequant_;
  phi::DenseTensor &workspace_;        // char
  phi::DenseTensor &input_workspace_;  // int8_t
  phi::DenseTensor &out_workspace_;    // int32_t

  std::unique_ptr<CublasLtHelper<int32_t>> cublaslt_helper;
};

template <typename T>
class LtGEMMHelper {
 public:
  LtGEMMHelper(
      const phi::GPUContext &dev_ctx, int m, int k, int n, bool transpose_y)
      : dev_ctx_(dev_ctx), m_(m), k_(k), n_(n) {
    cublaslt_helper = std::make_unique<CublasLtHelper<T, float>>(
        m, k, n, dev_ctx.cublaslt_handle(), transpose_y);
  }

  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,
               phi::DenseTensor *output) {
    cublaslt_helper->GEMM(input->data<T>(),
                          weight->data<T>(),
                          output->data<T>(),
                          dev_ctx_.stream(),
                          nullptr,
                          0);
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int m_;
  int k_;
  int n_;

  std::unique_ptr<CublasLtHelper<T, float>> cublaslt_helper;
};

template <typename T>
class NormHelper {
 public:
  NormHelper(const phi::GPUContext &dev_ctx,
             const std::string &norm_type,
             const int rows,
             const int cols,
             const float epsilon,
             const float residual_alpha)
      : dev_ctx_(dev_ctx),
        norm_type_(norm_type),
        rows_(rows),
        cols_(cols),
        epsilon_(epsilon),
        residual_alpha_(
            residual_alpha),  // TODO(zhengzekang): currently only available for
                              // Layernorm. Need support rmsnorm.
        layernorm_helper_(dev_ctx_, epsilon_, rows_, cols_) {
    // VLOG(0) << "NormHelper residual_alpha:" << residual_alpha_;
    DropoutParam dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    residual_bias_add_layernorm_helper_ =
        FusedDropoutLayerNormHelper<T, uint8_t>(
            dev_ctx, rows_, cols_, dropout_param, epsilon_, residual_alpha_);
  }

  /*
  Note(Zhengzekang):
  Since input `X` and `Residual` in FusedMT will be swaped by preallocated
  buffer, I have no choice but to pass the data pointer instead of
  phi::DenseTensor.
  */

  // dst = Norm(x + residual + bias(optional))
  void NormResidualBias(const T *x_data,
                        const T *residual_data,
                        const phi::DenseTensor *bias,
                        const phi::DenseTensor *norm_weight,
                        const phi::DenseTensor *norm_bias,
                        phi::DenseTensor *mean,
                        phi::DenseTensor *var,
                        phi::DenseTensor *bias_residual_out,
                        phi::DenseTensor *output) {
    using U = LayerNormParamType<T>;
    const T *bias_data = bias ? bias->data<T>() : nullptr;
    U *mean_data = mean ? mean->data<U>() : nullptr;
    U *var_data = var ? var->data<U>() : nullptr;
    T *bias_residual_out_data = bias_residual_out->data<T>();
    T *output_data = output->data<T>();

    if (norm_type_ == "layernorm") {
      // For layernorm, it use FP32 type weight and bias.
      const U *norm_weight_data =
          norm_weight ? norm_weight->data<U>() : nullptr;
      const U *norm_bias_data = norm_bias ? norm_bias->data<U>() : nullptr;
      residual_bias_add_layernorm_helper_.LayernormResidualDropoutBias(
          dev_ctx_,
          x_data,
          residual_data,
          bias_data,
          norm_weight_data,
          norm_bias_data,
          bias_residual_out_data,
          nullptr,
          output_data,
          mean_data,
          var_data);
      // } else if (norm_type_ == "rmsnorm") {
      //   // For rmsnorm, it use Input's type weight and bias.
      //   const T *norm_weight_data =
      //       norm_weight ? norm_weight->data<T>() : nullptr;
      //   const T *norm_bias_data = norm_bias ? norm_bias->data<T>() : nullptr;
      //   phi::ResidualAddRmsNormWrapper<T, phi::GPUContext>(dev_ctx_,
      //                                                      x_data,
      //                                                      residual_data,
      //                                                      bias_data,
      //                                                      norm_weight_data,
      //                                                      norm_bias_data,
      //                                                      epsilon_,
      //                                                      rows_,
      //                                                      cols_,
      //                                                      bias_residual_out_data,
      //                                                      output_data);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently NormHelper only support `layernorm`, `rmsnorm`. "));
    }
  }

  // dst = Norm(x)
  void Norm(const T *x_data,
            const phi::DenseTensor *norm_weight,
            const phi::DenseTensor *norm_bias,
            phi::DenseTensor *mean,
            phi::DenseTensor *var,
            phi::DenseTensor *output) {
    using U = LayerNormParamType<T>;
    U *mean_data = mean ? mean->data<U>() : nullptr;
    U *var_data = var ? var->data<U>() : nullptr;
    T *output_data = output->data<T>();

    if (norm_type_ == "layernorm") {
      // For layernorm, it use FP32 type weight and bias.
      const U *norm_weight_data =
          norm_weight ? norm_weight->data<U>() : nullptr;
      const U *norm_bias_data = norm_bias ? norm_bias->data<U>() : nullptr;
      layernorm_helper_.ComputeForward(x_data,
                                       norm_weight_data,
                                       norm_bias_data,
                                       output_data,
                                       mean_data,
                                       var_data);
      // } else if (norm_type_ == "rmsnorm") {
      //   // For rmsnorm, it use Input's type weight and bias.
      //   const T *norm_weight_data =
      //       norm_weight ? norm_weight->data<T>() : nullptr;
      //   const T *norm_bias_data = norm_bias ? norm_bias->data<T>() : nullptr;
      //   phi::RmsNormWrapper<T, phi::GPUContext>(dev_ctx_,
      //                                           x_data,
      //                                           norm_weight_data,
      //                                           norm_bias_data,
      //                                           epsilon_,
      //                                           rows_,
      //                                           cols_,
      //                                           output_data);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently NormHelper only support `layernorm`, `rmsnorm`. "));
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  std::string norm_type_;
  int rows_;
  int cols_;
  float epsilon_;
  float residual_alpha_;
  FusedDropoutLayerNormHelper<T, uint8_t> residual_bias_add_layernorm_helper_;
  AttnLayerNorm<T> layernorm_helper_;
};

template <typename T, typename nvT = typename PDDataTypeTraits<T>::DataType>
class FFNHelper {
 public:
  FFNHelper(const phi::GPUContext &dev_ctx,
            const std::string &act_method,
            int token_num,
            int dim_ffn,
            int dim_embed,
            const std::string gemm_method)
      : dev_ctx_(dev_ctx),
        act_method_(act_method),
        token_num_(token_num),
        dim_ffn_(dim_ffn),
        dim_embed_(dim_embed),
        gemm_method_(gemm_method) {}

  // dst = act(fc(src[0]) + bias) * src[1]
  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,
               const phi::DenseTensor *scale,
               const phi::DenseTensor *bias,
               phi::DenseTensor *workspace,
               phi::DenseTensor *bias_out,
               phi::DenseTensor *output) {
    /*
    input's shape [token_num, dim_embed]
    weight's shape [dim_embed, dim_ffn]
    bias' shape [dim_ffn]
    output's shape [token_num, dim_ffn].
    */
    // for debug
    VLOG(5) << "FFNHelper,"
            << " token_num_:" << token_num_ << " dim_ffn_:" << dim_ffn_
            << " dim_embed_:" << dim_embed_;
    GEMMHelper<T, nvT> gemm_helper(
        dev_ctx_, token_num_, dim_ffn_, dim_embed_, gemm_method_);
    BiasActHelper<T> bias_act_helper(
        dev_ctx_, act_method_, token_num_, dim_ffn_);

    gemm_helper.Compute(input, weight, scale, bias, workspace, bias_out);
    if (gemm_method_ == "LLm.int8") {
      bias_act_helper.Compute(bias_out, bias, output);
    } else {
      // Note(Zhengzekang): Other Gemm method can fuse bias add.
      bias_act_helper.Compute(bias_out, nullptr, output);
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  std::string act_method_;
  int token_num_;
  int dim_ffn_;
  int dim_embed_;
  std::string gemm_method_;
};


template <typename T>
class WriteCacheKVHelper {
 public:
  WriteCacheKVHelper(const phi::GPUContext &dev_ctx,
                     int quant_round_type,
                     float quant_max_bound,
                     float quant_min_bound)
      : dev_ctx_(dev_ctx),
        quant_round_type_(quant_round_type),
        quant_min_bound_(quant_min_bound),
        quant_max_bound_(quant_max_bound) {}

  void Compute(const phi::DenseTensor *pre_cache_kv_out,
               phi::DenseTensor *cache_kv_out,
               const phi::DenseTensor *kv_transpose_out,
               const int *sequence_lengths_data,
               const int cache_bsz,
               const int bsz,
               const int num_head,
               const int seq_len,
               const int dim_head,
               const int cache_offset,
               const float cache_k_scale,
               const float cache_v_scale) {
    if (cache_k_scale > 0) {
      WriteInt8CacheKV<T>(dev_ctx_,
                          pre_cache_kv_out,
                          cache_kv_out,
                          kv_transpose_out,
                          sequence_lengths_data,
                          cache_bsz,
                          bsz,
                          num_head,
                          seq_len,
                          dim_head,
                          cache_offset,
                          quant_round_type_,
                          quant_max_bound_,
                          quant_min_bound_,
                          cache_k_scale,
                          cache_v_scale);
    } else {
      WriteCacheKV<T>(dev_ctx_,
                      pre_cache_kv_out,
                      cache_kv_out,
                      kv_transpose_out,
                      sequence_lengths_data,
                      cache_bsz,
                      bsz,
                      num_head,
                      seq_len,
                      dim_head,
                      cache_offset);
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int quant_round_type_;
  float quant_max_bound_;
  float quant_min_bound_;
};

template <typename T>
class AttnOutHelper {
 public:
  AttnOutHelper(const phi::GPUContext &dev_ctx,
                phi::DenseTensor &workspace,          // NOLINT
                phi::DenseTensor &tmp_quant_space,    // NOLINT
                phi::DenseTensor &tmp_dequant_space,  // NOLINT
                int token_num,                        // m
                int hidden_size,                      // k
                int dim_embed,                        // n
                int ring_id,
                CustomNCCLComm *nccl_comm,
                int quant_round_type,
                float quant_max_bound,
                float quant_min_bound,
                bool is_decoder)
      : dev_ctx_(dev_ctx),
        token_num_(token_num),
        hidden_size_(hidden_size),
        dim_embed_(dim_embed),
        ring_id_(ring_id),
        nccl_comm_(nccl_comm),
        quant_round_type_(quant_round_type),
        quant_min_bound_(quant_min_bound),
        quant_max_bound_(quant_max_bound),
        workspace_(workspace),
        tmp_quant_space_(tmp_quant_space),
        tmp_dequant_space_(tmp_dequant_space),
        is_decoder_(is_decoder) {
    int8_gemm_helper_ = std::make_unique<Int8GEMMHelper<T>>(
        dev_ctx_,
        token_num,
        hidden_size,
        dim_embed,
        workspace,
        tmp_quant_space,
        tmp_dequant_space,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        is_decoder && FLAGS_use_gemm_dequant /*use_gemm_dequant*/);
    gemm_helper_ = std::make_unique<LtGEMMHelper<T>>(
        dev_ctx_, token_num, hidden_size, dim_embed, false);
  }

  void Compute(const phi::DenseTensor &input,
               const phi::DenseTensor &weight,
               const phi::DenseTensor &dequant_out_scales,
               const float in_scale,
               phi::DenseTensor *output) {
    if (nccl_comm_) {
      nccl_comm_->SwapInput(output);
    }
    if (in_scale > 0) {
      int8_gemm_helper_->Compute(&input,   // T
                                 &weight,  // int8, Need be transposed
                                 &dequant_out_scales,
                                 in_scale,
                                 output,        // T
                                 !is_decoder_,  // quant in mmha in decoder
                                 true);  // need to dequant cause allreduce
    } else {
      gemm_helper_->Compute(&input, &weight, output);
    }
    if (nccl_comm_) {
      *output = nccl_comm_->AllReduce();
    } else {
      AllReduce<T>(*output, ring_id_, output->numel(), dev_ctx_);
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int token_num_;    // m
  int hidden_size_;  // k
  int dim_embed_;    // n
  int ring_id_;
  int quant_round_type_;
  float quant_max_bound_;
  float quant_min_bound_;
  bool is_decoder_;
  CustomNCCLComm *nccl_comm_;
  phi::DenseTensor &workspace_;          // char
  phi::DenseTensor &tmp_quant_space_;    // int8_t
  phi::DenseTensor &tmp_dequant_space_;  // int32_t
  std::unique_ptr<Int8GEMMHelper<T>> int8_gemm_helper_;
  std::unique_ptr<LtGEMMHelper<T>> gemm_helper_;
};


}  // namespace

}  // namespace operators
}  // namespace paddle
