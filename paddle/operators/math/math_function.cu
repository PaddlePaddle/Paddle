/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <>
void gemm<platform::GPUPlace, float>(const platform::DeviceContext& context,
                                     const CBLAS_TRANSPOSE transA,
                                     const CBLAS_TRANSPOSE transB, const int M,
                                     const int N, const int K,
                                     const float alpha, const float* A,
                                     const float* B, const float beta,
                                     float* C) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gemm<platform::GPUPlace, double>(const platform::DeviceContext& context,
                                      const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB, const int M,
                                      const int N, const int K,
                                      const double alpha, const double* A,
                                      const double* B, const double beta,
                                      double* C) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void gemm<platform::GPUPlace, float>(const platform::DeviceContext& context,
                                     const bool transA, const bool transB,
                                     const int M, const int N, const int K,
                                     const float alpha, const float* A,
                                     const int lda, const float* B,
                                     const int ldb, const float beta, float* C,
                                     const int ldc) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = transB == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

template <>
void gemm<platform::GPUPlace, double>(const platform::DeviceContext& context,
                                      const bool transA, const bool transB,
                                      const int M, const int N, const int K,
                                      const double alpha, const double* A,
                                      const int lda, const double* B,
                                      const int ldb, const double beta,
                                      double* C, const int ldc) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = transB == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<const platform::CUDADeviceContext&>(context)
          .cublas_handle(),
      cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));
}

template <>
void matmul<platform::GPUPlace, float>(
    const platform::DeviceContext& context, const framework::Tensor& matrix_a,
    bool trans_a, const framework::Tensor& matrix_b, bool trans_b, float alpha,
    framework::Tensor* matrix_out, float beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in GPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::GPUPlace, float>(
      context, transA, transB, M, N, K, alpha, matrix_a.data<float>(),
      matrix_b.data<float>(), beta, matrix_out->data<float>());
}

template <>
void matmul<platform::GPUPlace, double>(
    const platform::DeviceContext& context, const framework::Tensor& matrix_a,
    bool trans_a, const framework::Tensor& matrix_b, bool trans_b, double alpha,
    framework::Tensor* matrix_out, double beta) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_gpu_place(matrix_a.place()) &&
                     platform::is_gpu_place(matrix_b.place()) &&
                     platform::is_gpu_place(matrix_out->place()),
                 "Matrix must all be in GPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::GPUPlace, double>(
      context, transA, transB, M, N, K, alpha, matrix_a.data<double>(),
      matrix_b.data<double>(), beta, matrix_out->data<double>());
}

template <typename T>
struct SelectedRowsAdd<platform::GPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::SelectedRows& input2,
                  framework::SelectedRows* output) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(in1_height, input2.height());
    output->set_height(in1_height);

    auto& in1_rows = input1.rows();
    auto& in2_rows = input2.rows();
    std::vector<int64_t> out_rows;
    out_rows.reserve(in1_rows.size() + in2_rows.size());

    // concat rows
    out_rows.insert(out_rows.end(), in1_rows.begin(), in1_rows.end());
    out_rows.insert(out_rows.end(), in2_rows.begin(), in2_rows.end());
    output->set_rows(out_rows);

    auto* out_value = output->mutable_value();
    auto& in1_value = input1.value();
    auto& in2_value = input2.value();

    auto in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, in2_value.numel() / in2_rows.size());
    PADDLE_ENFORCE_EQ(in1_row_numel, out_value->numel() / out_rows.size());

    auto* out_data = out_value->data<T>();
    auto* in1_data = in1_value.data<T>();

    auto in1_place = input1.place();
    PADDLE_ENFORCE(platform::is_gpu_place(in1_place));
    auto in2_place = input2.place();
    PADDLE_ENFORCE(platform::is_gpu_place(in2_place));
    auto out_place = context.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(out_place))

    memory::Copy(
        boost::get<platform::GPUPlace>(out_place), out_data,
        boost::get<platform::GPUPlace>(in1_place), in1_data,
        in1_value.numel() * sizeof(T),
        reinterpret_cast<const platform::CUDADeviceContext&>(context).stream());

    auto* in2_data = in2_value.data<T>();
    memory::Copy(
        boost::get<platform::GPUPlace>(out_place), out_data + in1_value.numel(),
        boost::get<platform::GPUPlace>(in2_place), in2_data,
        in2_value.numel() * sizeof(T),
        reinterpret_cast<const platform::CUDADeviceContext&>(context).stream());
  }
};

template struct SelectedRowsAdd<platform::GPUPlace, float>;

namespace {
template <int block_size, typename T>
__global__ void SelectedRowsAddTensorKernel(T* selected_rows, int64_t* rows,
                                            T* tensor_in, T* tensor_out,
                                            const int64_t row_numel) {
  const ty = blockIdx.y;
  int tid = threadIdx.x;

  selected_rows += ty * row_numel;
  tensor_in += rows[ty] * row_numel;
  tensor_out += rows[ty] * row_numel;

  for (int index = tid; index < row_numel; index += block_size) {
    tensor_out[index] = tensor_in[index] + selected_rows[index];
  }
}
}

template <typename T>
struct SelectedRowsAddTensor<platform::GPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::Tensor& input2, framework::Tensor* output) {
    auto in1_height = input1.height();
    auto in2_dims = input2.dims();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_EQ(in1_height, in2_dims[0]);
    PADDLE_ENFORCE_EQ(in1_height, out_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, input2.numel() / in1_height);
    PADDLE_ENFORCE_EQ(in1_row_numel, output->numel() / in1_height);

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = input2.data<T>();
    auto* out_data = output->data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid(1, in1_height);
    SelectedRowsAddTensorKernel<block_size, T><<<
        grid, threads, 0,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
        in1_data, in1_rows.data(), in2_data, out_data, in1_row_numel);
  }
};

template struct SelectedRowsAddTensor<platform::GPUPlace, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
