/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_gemm_epilogue_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedGemmEpilogueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");

    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    phi::DenseTensor* reserve_space =
        ctx.Output<phi::DenseTensor>("ReserveSpace");

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    std::string activation = ctx.Attr<std::string>("activation");
    VLOG(10) << "trans_x = " << trans_x << " , trans_y = " << trans_y
             << " , activation = " << activation;
    bool enable_auxiliary = reserve_space == nullptr ? false : true;

    dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));
    auto* out_data = out->data<T>();

    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
    // (M * K) * (K * N)
    int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
    int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (std::is_same<T, paddle::platform::float16>::value) {
      mat_type = CUDA_R_16F;
    }
    if (std::is_same<T, platform::bfloat16>::value) {
      mat_type = CUDA_R_16BF;
    }
    if (std::is_same<T, double>::value) {
      mat_type = CUDA_R_64F;
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    cublasLtMatmulDesc_t operation_desc = NULL;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
        &operation_desc, compute_type, scale_type));
    cublasOperation_t transx = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transy = trans_y ? CUBLAS_OP_T : CUBLAS_OP_N;
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &transx,
            sizeof(transx)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &transy,
            sizeof(transy)));

    cublasLtEpilogue_t epiloque_func =
        get_epilogue_type_(activation, enable_auxiliary);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epiloque_func,
            sizeof(epiloque_func)));
    const T* bias_data = bias->data<T>();
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_data,
            sizeof(bias_data)));

    if (enable_auxiliary && activation != "none") {
      // Note (Ming Huang): The initialization of ReseveSpace is happened in the
      // dev_ctx.Alloc. Therefore, we set real date type up here.
      if (activation == "relu") {
        paddle::experimental::DataType rs_type =
            paddle::experimental::DataType::BOOL;
        size_t reserve_space_size =
            phi::product(reserve_space->dims()) * SizeOf(rs_type);
        dev_ctx.Alloc(reserve_space, rs_type, reserve_space_size);
      } else {
        size_t reserve_space_size =
            phi::product(reserve_space->dims()) * sizeof(T);
        dev_ctx.Alloc<T>(reserve_space, reserve_space_size);
      }

      void* aux_data = reserve_space->data();

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              operation_desc,
              CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
              &aux_data,
              sizeof(aux_data)));
      int64_t aux_ld = N;
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              operation_desc,
              CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
              &aux_ld,
              sizeof(aux_ld)));
    }

    cublasLtMatrixLayout_t x_desc = NULL, y_desc = NULL, out_desc = NULL;
    if (trans_x)
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &x_desc, mat_type, M, K, M));
    else
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &x_desc, mat_type, K, M, K));
    if (trans_y)
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &y_desc, mat_type, K, N, K));
    else
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &y_desc, mat_type, N, K, N));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &out_desc, mat_type, N, M, N));

    cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();
    // NOTE(zengjinle): I do not know whether the 4MB workspace size is
    // "enough". I just followed the settings from the NVIDIA MLPerf BERT code.
    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    cudaStream_t stream = dev_ctx.stream();
    memory::allocation::AllocationPtr workspace = memory::Alloc(
        dev_ctx.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    double alpha64 = 1.0, beta64 = 0.0;
    float alpha32 = 1.0f, beta32 = 0.0f;
    void *alpha = nullptr, *beta = nullptr;
    if (std::is_same<T, double>::value) {
      alpha = &alpha64;
      beta = &beta64;
    } else {
      alpha = &alpha32;
      beta = &beta32;
    }

    const auto* y_data = y->data<T>();
    const auto* x_data = x->data<T>();

    auto algo = GemmEpilogueAlgoCache::Instance().GetGemmAlgo(lt_handle,
                                                              operation_desc,
                                                              y_desc,
                                                              x_desc,
                                                              out_desc,
                                                              alpha,
                                                              beta,
                                                              y_data,
                                                              x_data,
                                                              out_data,
                                                              stream,
                                                              workspace->ptr(),
                                                              workspace_size);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmul(lt_handle,
                                          operation_desc,
                                          alpha,
                                          y_data,
                                          y_desc,
                                          x_data,
                                          x_desc,
                                          beta,
                                          out_data,
                                          out_desc,
                                          out_data,
                                          out_desc,
                                          algo,
                                          workspace->ptr(),
                                          workspace_size,
                                          stream));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescDestroy(operation_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutDestroy(y_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutDestroy(x_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutDestroy(out_desc));
  }

 private:
  static cublasLtEpilogue_t get_epilogue_type_(const std::string& activation,
                                               bool enable_auxiliary) {
    if (activation == "relu") {
      return enable_auxiliary ? CUBLASLT_EPILOGUE_RELU_AUX_BIAS
                              : CUBLASLT_EPILOGUE_RELU_BIAS;
    } else if (activation == "gelu") {
      return enable_auxiliary ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS
                              : CUBLASLT_EPILOGUE_GELU_BIAS;
    } else if (activation == "none") {
      return CUBLASLT_EPILOGUE_BIAS;
    } else {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation));
    }
  }
};

enum FusedGEMMGradInType { kDX = 0, kDY = 1, kDZ = 2 };

template <bool TransX, bool TransY>
struct FusedGEMMGradTrait;

template <>
struct FusedGEMMGradTrait<false, false> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<true, false> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradATrans = false;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<false, true> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradATrans = false;
  static constexpr auto kXGradBTrans = false;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = false;
};

template <>
struct FusedGEMMGradTrait<true, true> {
  static constexpr auto kXGradA = FusedGEMMGradInType::kDY;
  static constexpr auto kXGradB = FusedGEMMGradInType::kDZ;
  static constexpr auto kXGradATrans = true;
  static constexpr auto kXGradBTrans = true;

  static constexpr auto kYGradA = FusedGEMMGradInType::kDZ;
  static constexpr auto kYGradB = FusedGEMMGradInType::kDX;
  static constexpr auto kYGradATrans = true;
  static constexpr auto kYGradBTrans = true;
};

static constexpr auto BoolToCuBlasEnum(bool transpose) {
  return transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
}

template <typename DeviceContext, typename T>
class FusedGemmEpilogueGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool transpose_x = ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.Attr<bool>("trans_y");

    if (transpose_x) {
      if (transpose_y) {
        ComputeImpl<true, true>(ctx);
      } else {
        ComputeImpl<true, false>(ctx);
      }
    } else {
      if (transpose_y) {
        ComputeImpl<false, true>(ctx);
      } else {
        ComputeImpl<false, false>(ctx);
      }
    }
  }

 private:
  template <bool TransX, bool TransY>
  static void ComputeImpl(const framework::ExecutionContext& ctx) {
    using Trait = FusedGEMMGradTrait<TransX, TransY>;
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    const phi::DenseTensor* dout = ctx.Input<phi::DenseTensor>("DOut");
    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* reserve_space =
        ctx.Input<phi::DenseTensor>("ReserveSpace");

    phi::DenseTensor* dx = ctx.Output<phi::DenseTensor>("DX");
    phi::DenseTensor* dy = ctx.Output<phi::DenseTensor>("DY");
    phi::DenseTensor* dbias = ctx.Output<phi::DenseTensor>("DBias");

    std::string activation_grad = ctx.Attr<std::string>("activation_grad");

    VLOG(10) << "trans_x = " << TransX << " , trans_y = " << TransY
             << " , activation_grad = " << activation_grad;

    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), TransX ? 1 : x->dims().size() - 1);

    // (M * K) * (K * N)
    int64_t M = TransX ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = TransY ? y->dims()[1] : y->dims()[0];
    int64_t N = TransY ? y->dims()[0] : y->dims()[1];

    VLOG(10) << "M = " << M << " , K = " << K << " , N = " << N;

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (std::is_same<T, paddle::platform::float16>::value) {
      mat_type = CUDA_R_16F;
    }
    if (std::is_same<T, platform::bfloat16>::value) {
      mat_type = CUDA_R_16BF;
    }
    if (std::is_same<T, double>::value) {
      mat_type = CUDA_R_64F;
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();
    // NOTE(zengjinle): I do not know whether the 4MB workspace size is
    // "enough". I just followed the settings from the NVIDIA MLPerf BERT code.
    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    const cublasLtMatmulAlgo_t* algo = nullptr;
    cudaStream_t stream = dev_ctx.stream();

    double alpha64 = 1.0, beta64 = 0.0;
    float alpha32 = 1.0f, beta32 = 0.0f;
    void *alpha = nullptr, *beta = nullptr;
    if (std::is_same<T, double>::value) {
      alpha = &alpha64;
      beta = &beta64;
    } else {
      alpha = &alpha32;
      beta = &beta32;
    }

    cublasLtMatrixLayout_t dout_desc = nullptr, dout_trans_desc = nullptr;
    cublasLtMatrixLayout_t x_desc = nullptr, x_trans_desc = nullptr;
    cublasLtMatrixLayout_t y_desc = nullptr, y_trans_desc = nullptr;
    cublasLtMatrixLayout_t dx_desc = nullptr, dy_desc = nullptr;
    cublasLtMatmulDesc_t dx_operation_desc = nullptr,
                         dy_operation_desc = nullptr;

    DEFINE_PADDLE_SCOPE_GUARD([&] {
      auto descs = {dout_desc,
                    dout_trans_desc,
                    x_desc,
                    x_trans_desc,
                    y_desc,
                    y_trans_desc,
                    dx_desc,
                    dy_desc};
      for (auto desc : descs) {
        if (desc) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::cublasLtMatrixLayoutDestroy(desc));
        }
      }

      if (dx_operation_desc) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescDestroy(dx_operation_desc));
      }

      if (dy_operation_desc) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescDestroy(dy_operation_desc));
      }
    });

    auto x_row = TransX ? K : M;
    auto x_col = TransX ? M : K;
    auto y_row = TransY ? N : K;
    auto y_col = TransY ? K : N;
    auto z_row = TransX ? N : M;
    auto z_col = TransX ? M : N;

    // dx = func(dout, y)
    if (dx) {
      constexpr auto kXGradAIsDZ = (Trait::kXGradA == FusedGEMMGradInType::kDZ);
      cublasLtMatrixLayout_t *dx_dout_desc, *dx_y_desc;

      if (TransX) {
        dx_dout_desc = &dout_trans_desc;
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatrixLayoutCreate(
                dx_dout_desc, mat_type, z_row, z_col, z_row));
      } else {
        dx_dout_desc = &dout_desc;
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatrixLayoutCreate(
                dx_dout_desc, mat_type, z_col, z_row, z_col));
      }

      dx_y_desc = &y_trans_desc;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          dx_y_desc, mat_type, y_col, y_row, y_col));

      auto& a_desc = kXGradAIsDZ ? (*dx_dout_desc) : (*dx_y_desc);
      auto& b_desc = kXGradAIsDZ ? (*dx_y_desc) : (*dx_dout_desc);
      auto a_trans = BoolToCuBlasEnum(Trait::kXGradATrans);
      auto b_trans = BoolToCuBlasEnum(Trait::kXGradBTrans);

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &dx_desc, mat_type, x_col, x_row, x_col));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
          &dx_operation_desc, compute_type, scale_type));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dx_operation_desc,
              CUBLASLT_MATMUL_DESC_TRANSB,
              &a_trans,
              sizeof(a_trans)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dx_operation_desc,
              CUBLASLT_MATMUL_DESC_TRANSA,
              &b_trans,
              sizeof(b_trans)));

      cublasLtEpilogue_t epiloque_func_for_dx =
          get_epilogue_type_(activation_grad);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dx_operation_desc,
              CUBLASLT_MATMUL_DESC_EPILOGUE,
              &epiloque_func_for_dx,
              sizeof(epiloque_func_for_dx)));

      if (activation_grad != "none") {
        auto* aux_data = reserve_space->data();
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescSetAttribute(
                dx_operation_desc,
                CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                &aux_data,
                sizeof(aux_data)));
        int64_t aux_ld = TransX ? M : K;
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescSetAttribute(
                dx_operation_desc,
                CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                &aux_ld,
                sizeof(aux_ld)));
      }

      auto dx_workspace = memory::Alloc(
          dev_ctx.GetPlace(),
          workspace_size,
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

      auto* dx_data = dev_ctx.Alloc<T>(dx, dx->numel() * sizeof(T));
      const auto* y_data = y->data<T>();
      const auto* dout_data = dout->data<T>();
      const auto* a_data = kXGradAIsDZ ? dout_data : y_data;
      const auto* b_data = kXGradAIsDZ ? y_data : dout_data;

      auto algo =
          GemmEpilogueAlgoCache::Instance().GetGemmAlgo(lt_handle,
                                                        dx_operation_desc,
                                                        b_desc,
                                                        a_desc,
                                                        dx_desc,
                                                        alpha,
                                                        beta,
                                                        b_data,
                                                        a_data,
                                                        dx_data,
                                                        stream,
                                                        dx_workspace->ptr(),
                                                        workspace_size);

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmul(lt_handle,
                                            dx_operation_desc,
                                            alpha,
                                            b_data,
                                            b_desc,
                                            a_data,
                                            a_desc,
                                            beta,
                                            dx_data,
                                            dx_desc,
                                            dx_data,
                                            dx_desc,
                                            algo,
                                            dx_workspace->ptr(),
                                            workspace_size,
                                            stream));
    }

    // dy = func(dout, x)
    if (dy) {
      constexpr auto kYGradAIsDZ = (Trait::kYGradA == FusedGEMMGradInType::kDZ);

      cublasLtMatrixLayout_t *dy_dout_desc = nullptr, *dy_x_desc = nullptr;
      if (TransX) {
        dy_dout_desc = &dout_trans_desc;
        if (dout_trans_desc == nullptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::cublasLtMatrixLayoutCreate(
                  dy_dout_desc, mat_type, z_row, z_col, z_row));
        }
      } else {
        dy_dout_desc = &dout_desc;
        if (dout_desc == nullptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::cublasLtMatrixLayoutCreate(
                  dy_dout_desc, mat_type, z_col, z_row, z_col));
        }
      }

      dy_x_desc = &x_trans_desc;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          dy_x_desc, mat_type, x_col, x_row, x_col));

      auto& a_desc = kYGradAIsDZ ? (*dy_dout_desc) : (*dy_x_desc);
      auto& b_desc = kYGradAIsDZ ? (*dy_x_desc) : (*dy_dout_desc);
      auto a_trans = BoolToCuBlasEnum(Trait::kYGradATrans);
      auto b_trans = BoolToCuBlasEnum(Trait::kYGradBTrans);

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &dy_desc, mat_type, y_col, y_row, y_col));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
          &dy_operation_desc, compute_type, scale_type));

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dy_operation_desc,
              CUBLASLT_MATMUL_DESC_TRANSB,
              &a_trans,
              sizeof(a_trans)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dy_operation_desc,
              CUBLASLT_MATMUL_DESC_TRANSA,
              &b_trans,
              sizeof(b_trans)));

      cublasLtEpilogue_t epiloque_func_for_dy;
      if (dbias == nullptr) {
        epiloque_func_for_dy = CUBLASLT_EPILOGUE_DEFAULT;
      } else {
        if (TransY) {
          epiloque_func_for_dy = CUBLASLT_EPILOGUE_BGRADB;
        } else {
          epiloque_func_for_dy = CUBLASLT_EPILOGUE_BGRADA;
        }
      }

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dy_operation_desc,
              CUBLASLT_MATMUL_DESC_EPILOGUE,
              &epiloque_func_for_dy,
              sizeof(epiloque_func_for_dy)));

      if (dbias) {
        auto* dbias_data = dev_ctx.Alloc<T>(dbias, dbias->numel() * sizeof(T));
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescSetAttribute(
                dy_operation_desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &dbias_data,
                sizeof(dbias_data)));
      }

      auto dy_workspace = memory::Alloc(
          dev_ctx.GetPlace(),
          workspace_size,
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
      auto* dy_data = dev_ctx.Alloc<T>(dy, dy->numel() * sizeof(T));
      const auto* dout_data = dout->data<T>();
      const auto* x_data = x->data<T>();
      const auto* a_data = kYGradAIsDZ ? dout_data : x_data;
      const auto* b_data = kYGradAIsDZ ? x_data : dout_data;

      auto algo =
          GemmEpilogueAlgoCache::Instance().GetGemmAlgo(lt_handle,
                                                        dy_operation_desc,
                                                        b_desc,
                                                        a_desc,
                                                        dy_desc,
                                                        alpha,
                                                        beta,
                                                        b_data,
                                                        a_data,
                                                        dy_data,
                                                        stream,
                                                        dy_workspace->ptr(),
                                                        workspace_size);

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmul(lt_handle,
                                            dy_operation_desc,
                                            alpha,
                                            b_data,
                                            b_desc,
                                            a_data,
                                            a_desc,
                                            beta,
                                            dy_data,
                                            dy_desc,
                                            dy_data,
                                            dy_desc,
                                            algo,
                                            dy_workspace->ptr(),
                                            workspace_size,
                                            stream));
    }
  }

 private:
  static cublasLtEpilogue_t get_epilogue_type_(
      const std::string& activation_grad) {
    if (activation_grad == "relu_grad") {
      return CUBLASLT_EPILOGUE_DRELU;
    } else if (activation_grad == "gelu_grad") {
      return CUBLASLT_EPILOGUE_DGELU;
    } else if (activation_grad == "none") {
      return CUBLASLT_EPILOGUE_DEFAULT;
    } else {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::InvalidArgument(
              "The activation_grad attribute of fused_gemm_epilogue op should "
              "be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation_grad=%s.",
              activation_grad));
    }
  }
};

}  // namespace operators
}  // namespace paddle

#if CUDA_VERSION >= 11060
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_gemm_epilogue,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, float>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, double>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, paddle::platform::float16>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, paddle::platform::bfloat16>);

REGISTER_OP_CUDA_KERNEL(
    fused_gemm_epilogue_grad,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext, float>,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext, double>,
    ops::FusedGemmEpilogueGradKernel<phi::GPUContext,
                                     paddle::platform::float16>,
    ops::FusedGemmEpilogueKernel<phi::GPUContext, paddle::platform::bfloat16>);
#endif
