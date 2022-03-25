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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FusedGemmEpilogueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    const Tensor* bias = ctx.Input<Tensor>("Bias");

    Tensor* out = ctx.Output<Tensor>("Out");
    Tensor* reserve_space = ctx.Output<Tensor>("ReserveSpace");

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    std::string activation = ctx.Attr<std::string>("activation");
    bool enable_auxiliary = reserve_space == nullptr ? false : true;

    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), trans_x ? 1 : x->dims().size() - 1);
    int64_t M = trans_x ? x_mat_dims[1] : x_mat_dims[0];
    int64_t K = trans_y ? y->dims()[1] : y->dims()[0];
    int64_t N = trans_y ? y->dims()[0] : y->dims()[1];

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (std::is_same<T, paddle::platform::float16>::value) {
      mat_type = CUDA_R_16F;
      scale_type = CUDA_R_16F;
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
            operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transx,
            sizeof(transx)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transy,
            sizeof(transy)));

    cublasLtEpilogue_t epiloque_func =
        get_epilogue_type_(activation, enable_auxiliary);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epiloque_func,
            sizeof(epiloque_func)));
    const T* bias_data = bias->data<T>();
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_data,
            sizeof(bias_data)));

    if (enable_auxiliary && activation != "none") {
      size_t reserve_space_size = 0;
      if (activation == "relu") {
        // Count in bits.
        reserve_space_size = phi::product(out->dims()) / 8;
      } else {
        reserve_space_size = phi::product(out->dims()) * sizeof(T);
      }
      reserve_space->mutable_data(ctx.GetPlace(), out->type(),
                                  reserve_space_size);
      void* aux_data = reinterpret_cast<void*>(reserve_space->data<T>());

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
              &aux_data, sizeof(aux_data)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &N,
              sizeof(N)));
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
    size_t workspace_size = 4 * 1024 * 1024;
    const cublasLtMatmulAlgo_t* algo = nullptr;
    cudaStream_t stream = dev_ctx.stream();
    memory::allocation::AllocationPtr workspace =
        memory::Alloc(dev_ctx, workspace_size);

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

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmul(
        lt_handle, operation_desc, alpha, y->data<T>(), y_desc, x->data<T>(),
        x_desc, beta, out_data, out_desc, out_data, out_desc, algo,
        workspace->ptr(), workspace_size, stream));
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
          true, false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation));
    }
  }
};

template <typename DeviceContext, typename T>
class FusedGemmEpilogueGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    const Tensor* dout = ctx.Input<Tensor>("DOut");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* y = ctx.Input<Tensor>("Y");
    const Tensor* reserve_space = ctx.Input<Tensor>("ReserveSpace");

    Tensor* dx = ctx.Output<Tensor>("DX");
    Tensor* dy = ctx.Output<Tensor>("DY");
    Tensor* dbias = ctx.Output<Tensor>("DBias");

    std::string activation_grad = ctx.Attr<std::string>("activation_grad");

    auto dout_mat_dims =
        phi::flatten_to_2d(dout->dims(), dout->dims().size() - 1);
    auto x_mat_dims = phi::flatten_to_2d(x->dims(), x->dims().size() - 1);

    int64_t M = x_mat_dims[0];
    int64_t K = y->dims()[0];
    int64_t N = y->dims()[1];

    cudaDataType_t mat_type = CUDA_R_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (std::is_same<T, paddle::platform::float16>::value) {
      mat_type = CUDA_R_16F;
      scale_type = CUDA_R_16F;
    }
    if (std::is_same<T, double>::value) {
      mat_type = CUDA_R_64F;
      scale_type = CUDA_R_64F;
      compute_type = CUBLAS_COMPUTE_64F;
    }

    cublasLtHandle_t lt_handle = dev_ctx.cublaslt_handle();
    size_t workspace_size = 4 * 1024 * 1024;
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

    cublasOperation_t trans_dout = CUBLAS_OP_N;
    cublasLtMatrixLayout_t dout_desc = NULL;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &dout_desc, mat_type, N, M, N));

    if (dx) {
      cublasLtMatmulDesc_t dx_operation_desc = NULL;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
          &dx_operation_desc, compute_type, scale_type));
      cublasOperation_t trans_y = CUBLAS_OP_T;
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dx_operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_dout,
              sizeof(trans_dout)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dx_operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_y,
              sizeof(trans_y)));
      cublasLtEpilogue_t epiloque_func_for_dx =
          get_epilogue_type_(activation_grad);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dx_operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
              &epiloque_func_for_dx, sizeof(epiloque_func_for_dx)));

      if (activation_grad != "none") {
        auto* aux_data = reserve_space->data<T>();
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescSetAttribute(
                dx_operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                &aux_data, sizeof(aux_data)));
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescSetAttribute(
                dx_operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &N,
                sizeof(N)));
      }

      cublasLtMatrixLayout_t y_desc = NULL, dx_desc = NULL;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &y_desc, mat_type, N, K, N));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &dx_desc, mat_type, K, M, K));

      memory::allocation::AllocationPtr dx_workspace =
          memory::Alloc(dev_ctx, workspace_size);

      dx->mutable_data<T>(ctx.GetPlace());
      auto* dx_data = dx->data<T>();
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmul(
          lt_handle, dx_operation_desc, alpha, y->data<T>(), y_desc,
          dout->data<T>(), dout_desc, beta, dx_data, dx_desc, dx_data, dx_desc,
          algo, dx_workspace->ptr(), workspace_size, stream));
    }

    if (dy) {
      cublasLtMatmulDesc_t dy_operation_desc = NULL;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
          &dy_operation_desc, compute_type, scale_type));
      cublasOperation_t trans_x = CUBLAS_OP_T;
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dy_operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_dout,
              sizeof(trans_dout)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dy_operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_x,
              sizeof(trans_x)));
      cublasLtEpilogue_t epiloque_func_for_dy = dbias == nullptr
                                                    ? CUBLASLT_EPILOGUE_DEFAULT
                                                    : CUBLASLT_EPILOGUE_BGRADA;
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatmulDescSetAttribute(
              dy_operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
              &epiloque_func_for_dy, sizeof(epiloque_func_for_dy)));

      if (dbias) {
        dbias->mutable_data<T>(ctx.GetPlace());
        auto* dbias_data = dbias->data<T>();
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::cublasLtMatmulDescSetAttribute(
                dy_operation_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &dbias_data, sizeof(dbias_data)));
      }

      cublasLtMatrixLayout_t x_desc = NULL, dy_desc = NULL;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &x_desc, mat_type, K, M, K));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &dy_desc, mat_type, N, K, N));

      memory::allocation::AllocationPtr dy_workspace =
          memory::Alloc(dev_ctx, workspace_size);

      dy->mutable_data<T>(ctx.GetPlace());
      auto* dy_data = dy->data<T>();
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmul(
          lt_handle, dy_operation_desc, alpha, dout->data<T>(), dout_desc,
          x->data<T>(), x_desc, beta, dy_data, dy_desc, dy_data, dy_desc, algo,
          dy_workspace->ptr(), workspace_size, stream));
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
          true, false,
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
    ops::FusedGemmEpilogueKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FusedGemmEpilogueKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FusedGemmEpilogueKernel<paddle::platform::CUDADeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(
    fused_gemm_epilogue_grad,
    ops::FusedGemmEpilogueGradKernel<paddle::platform::CUDADeviceContext,
                                     float>,
    ops::FusedGemmEpilogueGradKernel<paddle::platform::CUDADeviceContext,
                                     double>,
    ops::FusedGemmEpilogueGradKernel<paddle::platform::CUDADeviceContext,
                                     paddle::platform::float16>);
#endif
