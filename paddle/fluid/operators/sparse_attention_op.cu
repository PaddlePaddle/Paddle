/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <math.h>
#include <limits>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/platform/dynload/cusparse.h"
#endif

namespace ops = paddle::operators;
namespace plf = paddle::platform;

namespace paddle {
namespace operators {

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask, T val,
                                                int width = warpSize) {
  return __shfl_xor_sync(mask, val, width);
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val = CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T max_val = CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, int BlockSize, int NnzBlockMax, bool KpMode = false,
          bool AttnMode = false>
__global__ void BlockSparseSoftmaxForward(T* softmax, const T* src, T scale,
                                          const T* kp_mask, const T* attn_mask,
                                          const int* layout_rowptr,
                                          const int* layout_colindex,
                                          int seq_len) {
  // current thread related info
  const int WarpSize = 32;
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  if (cur < seq_len) {
    const int cur_seqb = cur / BlockSize;
    const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

    T srcdata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];
    T attndata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];

    // read kp mask
    T datakp_mask = (KpMode == true) ? kp_mask[cur] : 0;

    // read tensor data, attn mask
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    const T* srcptr = src + layout_rowptr[cur_seqb];
    const T* attnptr = attn_mask + cur_seqb * seq_len;
    const int* colindex = layout_colindex + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        if (AttnMode == true) {
          if (std::abs(attnptr[colindex[xidx]]) <
              std::numeric_limits<T>::epsilon()) {
            srcdata[didx] =
                -std::numeric_limits<T>::infinity() * scale + datakp_mask;
          } else {
            srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
          }
        } else {
          srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
        }
      } else {
        srcdata[didx] = -std::numeric_limits<T>::infinity();
      }
    }

    // max value
    T max_value = srcdata[0];
    const int kIteration = (cur_nnzb * BlockSize + WarpSize - 1) / WarpSize;
#pragma unroll
    for (int it = 1; it < kIteration; ++it) {
      max_value = (max_value > srcdata[it]) ? max_value : srcdata[it];
    }
    WarpReduceMax<T, 1, WarpSize>(&max_value);

    // exp sum
    T sum = 0;
#pragma unroll
    for (int it = 0; it < kIteration; ++it) {
      srcdata[it] = std::exp(srcdata[it] - max_value);
      sum += srcdata[it];
    }
    WarpReduceSum<T, 1, WarpSize>(&sum);

    // compute softmax and write out
    T* softmaxptr = softmax + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        softmaxptr[xidx] = srcdata[didx] / sum;
      }
    }
  }
}

template <typename T, int BlockSize, int NnzBlockMax>
__global__ void BlockSparseSoftmaxBackward(T* dst, const T* grad, const T* src,
                                           T scale, const int* layout_rowptr,
                                           const int* layout_colindex,
                                           int seq_len) {
  // current thread related info
  const int WarpSize = 32;
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  if (cur < seq_len) {
    const int cur_seqb = cur / BlockSize;
    const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

    T srcdata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];
    T graddata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];

    // read tensor data, attn mask
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    const T* srcptr = src + layout_rowptr[cur_seqb];
    const T* gradptr = grad + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        srcdata[didx] = srcptr[xidx];
        graddata[didx] = gradptr[xidx];
      } else {
        srcdata[didx] = 0;
        graddata[didx] = 0;
      }
    }

    T sum = 0;
    const int kIteration = (cur_nnzb * BlockSize + WarpSize - 1) / WarpSize;
#pragma unroll
    for (int it = 0; it < kIteration; ++it) {
      sum += srcdata[it] * graddata[it];
    }
    WarpReduceSum<T, 1, WarpSize>(&sum);

    // compute softmax and write out
    T* dstptr = dst + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        dstptr[xidx] = scale * srcdata[didx] * (graddata[didx] - sum);
      }
    }
  }
}

using Tensor = framework::Tensor;
/*
input: sparse C in CSR format (num_rows,num_rows)
output: sparse C after softmax operation
*/
template <typename DeviceContext, typename T>
void SparseSoftmaxForward(const platform::CUDADeviceContext& ctx,
                          const Tensor* offset, const Tensor* columns,
                          Tensor* input, Tensor* output, const int blocksize,
                          const int num_rows, const int num_cols) {
  const int* offset_data = offset->data<int>();
  const int* columns_data = columns->data<int>();
  T* input_data = input->data<T>();
  T* output_data = output->data<T>();

  const int BlockSize = 1;
  dim3 blocks(32, 4, 1);
  int grid = (num_rows * BlockSize + 3) / 4;
  T scaling = static_cast<T>(1.0) / sqrt(static_cast<double>(num_cols));

  const int NnzBlockMax = 256;
  BlockSparseSoftmaxForward<T, BlockSize, NnzBlockMax><<<grid, blocks>>>(
      output_data, input_data, scaling, NULL, NULL, offset_data, columns_data,
      num_rows);
}

template <typename DeviceContext, typename T>
void SparseSoftmaxBackward(const platform::CUDADeviceContext& ctx,
                           const Tensor* offset, const Tensor* columns,
                           Tensor* dx, const Tensor* dout, const Tensor* out,
                           const int blocksize, const int num_rows,
                           const int num_cols) {
  const int* offset_data = offset->data<int>();
  const int* columns_data = columns->data<int>();
  T* dx_data = dx->data<T>();
  const T* dout_data = dout->data<T>();
  const T* out_data = out->data<T>();

  const int BlockSize = 1;
  dim3 blocks(32, 4, 1);
  int grid = (num_rows * BlockSize + 3) / 4;
  T scaling = static_cast<T>(1.0) / sqrt(static_cast<double>(num_cols));

  const int NnzBlockMax = 256;
  BlockSparseSoftmaxBackward<T, BlockSize, NnzBlockMax><<<grid, blocks>>>(
      dx_data, dout_data, out_data, scaling, offset_data, columns_data,
      num_rows);
}

int GetGpuType(std::string data_type) {
  int gpu_type = -1;
  if (data_type == "float") {
    gpu_type = 0;
  } else if (data_type == "double") {
    gpu_type = 1;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Not support tensor type in sparse_attention OP: %s", data_type));
  }
  return gpu_type;
}

/*
input: dense A (num_rows,num_cols), dense B (num_rows,num_cols)
output: sparse C in CSR format (num_rows,num_rows)
*/
template <typename DeviceContext, typename T>
void DotSdd(const platform::CUDADeviceContext& ctx, const Tensor* A,
            const Tensor* B, const Tensor* C_offset, const Tensor* C_columns,
            Tensor* C_value, const int num_rows, const int num_cols,
            const bool A_transpose, const bool B_transpose) {
  const T* A_data = A->data<T>();
  const T* B_data = B->data<T>();
  const int* C_offset_data = C_offset->data<int>();
  const int* C_columns_data = C_columns->data<int>();
  T* C_value_data = C_value->data<T>();

  std::string data_type = framework::DataTypeToString(C_value->type());
  int gpu_type = GetGpuType(data_type);

  cusparseHandle_t handle = NULL;
  cusparseDnMatDescr_t matA, matB;
  cusparseSpMatDescr_t matC;
  platform::dynload::cusparseCreate(&handle);

  // Create dense matrix A
  platform::dynload::cusparseCreateDnMat(
      &matA, num_rows, num_cols, num_cols, const_cast<T*>(A_data),
      gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F, CUSPARSE_ORDER_ROW);
  // Create dense matrix B
  platform::dynload::cusparseCreateDnMat(
      &matB, num_rows, num_cols, num_cols, const_cast<T*>(B_data),
      gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F, CUSPARSE_ORDER_ROW);
  // Create sparse matrix C in CSR format
  int C_nnz = C_columns->dims()[1];
  platform::dynload::cusparseCreateCsr(
      &matC, num_rows, num_rows, C_nnz, const_cast<int*>(C_offset_data),
      const_cast<int*>(C_columns_data), C_value_data, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F);

  T alpha = 1;
  T beta = 0;

  void* dBuffer = NULL;
  size_t bufferSize = 0;
  platform::dynload::cusparseSDDMM_bufferSize(
      handle, A_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                          : CUSPARSE_OPERATION_NON_TRANSPOSE,
      B_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                  : CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC, gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F,
      CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  platform::dynload::cusparseSDDMM(
      handle, A_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                          : CUSPARSE_OPERATION_NON_TRANSPOSE,
      B_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                  : CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC, gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F,
      CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

  platform::dynload::cusparseDestroyDnMat(matA);
  platform::dynload::cusparseDestroyDnMat(matB);
  platform::dynload::cusparseDestroySpMat(matC);
  platform::dynload::cusparseDestroy(handle);
  cudaFree(dBuffer);
}

/*
input: sparse A in CSR format (num_rows,num_rows), dense B (num_rows,num_cols)
output: dense C (num_rows,num_cols)
*/
template <typename DeviceContext, typename T>
void DotDsd(const platform::CUDADeviceContext& ctx, const Tensor* A_offset,
            const Tensor* A_columns, const Tensor* A_value, const Tensor* B,
            Tensor* C, const int num_rows, const int num_cols,
            const bool A_transpose, const bool B_transpose) {
  const int* A_offset_data = A_offset->data<int>();
  const int* A_columns_data = A_columns->data<int>();
  const T* A_value_data = A_value->data<T>();
  const T* B_data = B->data<T>();
  T* C_data = C->data<T>();

  std::string data_type = framework::DataTypeToString(C->type());
  int gpu_type = GetGpuType(data_type);

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  platform::dynload::cusparseCreate(&handle);

  // Create sparse matrix A in CSR format
  int A_nnz = A_columns->dims()[1];
  platform::dynload::cusparseCreateCsr(
      &matA, num_rows, num_rows, A_nnz, const_cast<int*>(A_offset_data),
      const_cast<int*>(A_columns_data), const_cast<T*>(A_value_data),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F);

  // Create dense matrix B
  platform::dynload::cusparseCreateDnMat(
      &matB, num_rows, num_cols, num_cols, const_cast<T*>(B_data),
      gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F, CUSPARSE_ORDER_ROW);
  // Create dense matrix C
  platform::dynload::cusparseCreateDnMat(
      &matC, num_rows, num_cols, num_cols, C_data,
      gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F, CUSPARSE_ORDER_ROW);

  T alpha = 1;
  T beta = 0;

  void* dBuffer = NULL;
  size_t bufferSize = 0;
  // allocate an external buffer if needed
  platform::dynload::cusparseSpMM_bufferSize(
      handle, A_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                          : CUSPARSE_OPERATION_NON_TRANSPOSE,
      B_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                  : CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC, gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F,
      CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  platform::dynload::cusparseSpMM(
      handle, A_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                          : CUSPARSE_OPERATION_NON_TRANSPOSE,
      B_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                  : CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC, gpu_type == 0 ? CUDA_R_32F : CUDA_R_64F,
      CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

  platform::dynload::cusparseDestroySpMat(matA);
  platform::dynload::cusparseDestroyDnMat(matB);
  platform::dynload::cusparseDestroyDnMat(matC);
  platform::dynload::cusparseDestroy(handle);
  cudaFree(dBuffer);
}

std::vector<Tensor> GetSplitTensor(Tensor* input) {
  auto dims = input->dims();
  int batch_size = dims[0];
  int num_heads = dims[1];
  std::vector<int> NewDims(dims.size() - 1);
  NewDims[0] = batch_size * num_heads;
  for (int i = 1; i < NewDims.size(); i++) {
    NewDims[i] = dims[i + 1];
  }
  input->Resize(framework::make_ddim(NewDims));
  return input->Split(1, 0);
}
#endif

template <typename DeviceContext, typename T>
class SparseAttentionCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
    auto query = *ctx.Input<Tensor>("Q");
    auto key = *ctx.Input<Tensor>("K");
    auto value = *ctx.Input<Tensor>("V");
    auto offset = *ctx.Input<Tensor>("offset");
    auto columns = *ctx.Input<Tensor>("columns");
    auto outputPtr = ctx.Output<Tensor>("Out");
    outputPtr->mutable_data<T>(ctx.GetPlace());
    auto ResultSddPtr = ctx.Output<Tensor>("ResultSdd");
    ResultSddPtr->mutable_data<T>(ctx.GetPlace());
    auto ResultSoftmaxPtr = ctx.Output<Tensor>("ResultSoftmax");
    ResultSoftmaxPtr->mutable_data<T>(ctx.GetPlace());

    auto output = *outputPtr;
    auto result_sdd = *ResultSddPtr;
    auto result_softmax = *ResultSoftmaxPtr;

    auto query_dims = query.dims();
    int batch_size = query_dims[0];
    int num_heads = query_dims[1];
    int M = query_dims[2];
    int N = query_dims[3];

    std::vector<Tensor> query_lists = GetSplitTensor(&query);
    std::vector<Tensor> key_lists = GetSplitTensor(&key);
    std::vector<Tensor> value_lists = GetSplitTensor(&value);
    std::vector<Tensor> offset_lists = GetSplitTensor(&offset);
    std::vector<Tensor> columns_lists = GetSplitTensor(&columns);
    std::vector<Tensor> result_sdd_lists = GetSplitTensor(&result_sdd);
    std::vector<Tensor> result_softmax_lists = GetSplitTensor(&result_softmax);
    std::vector<Tensor> output_lists = GetSplitTensor(&output);

    const auto& dev_ctx = ctx.cuda_device_context();
    const int iter_num = batch_size * num_heads;
    for (int i = 0; i < iter_num; i++) {
      DotSdd<DeviceContext, T>(dev_ctx, &query_lists[i], &key_lists[i],
                               &offset_lists[i], &columns_lists[i],
                               &result_sdd_lists[i], M, N, false, true);

      SparseSoftmaxForward<DeviceContext, T>(
          dev_ctx, &offset_lists[i], &columns_lists[i], &result_sdd_lists[i],
          &result_softmax_lists[i], 1, M, N);

      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &result_softmax_lists[i], &value_lists[i],
                               &output_lists[i], M, N, false, false);
    }
#else
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The sparse_attention OP needs to use Nvidia GPU, and the CUDA version "
        "cannot be less than 11.2"));
#endif
  }
};

template <typename DeviceContext, typename T>
class SparseAttentionGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
    auto query = *ctx.Input<Tensor>("Q");
    auto key = *ctx.Input<Tensor>("K");
    auto value = *ctx.Input<Tensor>("V");
    auto offset = *ctx.Input<Tensor>("offset");
    auto columns = *ctx.Input<Tensor>("columns");
    auto ResultSdd = *ctx.Input<Tensor>("ResultSdd");
    auto ResultSoftmax = *ctx.Input<Tensor>("ResultSoftmax");
    auto dout = *ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dQueryPtr = ctx.Output<Tensor>(framework::GradVarName("Q"));
    auto* dKeyPtr = ctx.Output<Tensor>(framework::GradVarName("K"));
    auto* dValuePtr = ctx.Output<Tensor>(framework::GradVarName("V"));
    dQueryPtr->mutable_data<T>(ctx.GetPlace());
    dKeyPtr->mutable_data<T>(ctx.GetPlace());
    dValuePtr->mutable_data<T>(ctx.GetPlace());
    auto dQuery = *dQueryPtr;
    auto dKey = *dKeyPtr;
    auto dValue = *dValuePtr;

    auto query_dims = query.dims();
    int batch_size = query_dims[0];
    int num_heads = query_dims[1];
    int M = query_dims[2];
    int N = query_dims[3];

    std::vector<Tensor> query_lists = GetSplitTensor(&query);
    std::vector<Tensor> key_lists = GetSplitTensor(&key);
    std::vector<Tensor> value_lists = GetSplitTensor(&value);
    std::vector<Tensor> offset_lists = GetSplitTensor(&offset);
    std::vector<Tensor> columns_lists = GetSplitTensor(&columns);
    std::vector<Tensor> ResultSdd_lists = GetSplitTensor(&ResultSdd);
    std::vector<Tensor> ResultSoftmax_lists = GetSplitTensor(&ResultSoftmax);
    std::vector<Tensor> dout_lists = GetSplitTensor(&dout);
    std::vector<Tensor> dQuery_lists = GetSplitTensor(&dQuery);
    std::vector<Tensor> dKey_lists = GetSplitTensor(&dKey);
    std::vector<Tensor> dValue_lists = GetSplitTensor(&dValue);

    const int iter_num = batch_size * num_heads;
    const auto& dev_ctx = ctx.cuda_device_context();
    for (int i = 0; i < iter_num; i++) {
      // dValue = transpose(result_softmax) * dOut
      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &ResultSoftmax_lists[i], &dout_lists[i],
                               &dValue_lists[i], M, N, true, false);

      // dResultSoftmax = dOut * transpose(Value)
      int nnz_num = columns.dims()[0];
      Tensor dResultSoftmax;
      dResultSoftmax.Resize({nnz_num});
      dResultSoftmax.mutable_data<T>(ctx.GetPlace());
      DotSdd<DeviceContext, T>(dev_ctx, &dout_lists[i], &value_lists[i],
                               &offset_lists[i], &columns_lists[i],
                               &dResultSoftmax, M, N, false, true);

      // dResultSdd = dResultSoftmax * softmax'(ResultSdd)
      Tensor dResultSdd;
      dResultSdd.Resize({nnz_num});
      dResultSdd.mutable_data<T>(ctx.GetPlace());
      SparseSoftmaxBackward<DeviceContext, T>(
          dev_ctx, &offset_lists[i], &columns_lists[i], &dResultSdd,
          &dResultSoftmax, &ResultSoftmax_lists[i], 1, M, N);

      // dQuery = dResultSdd * Key
      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &dResultSdd, &key_lists[i], &dQuery_lists[i], M,
                               N, false, false);

      // dKey = transpose(dResultSdd) * Query
      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &dResultSdd, &query_lists[i], &dKey_lists[i], M,
                               N, true, false);
    }
#else
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The sparse_attention OP needs to use Nvidia GPU, and the CUDA version "
        "cannot be less than 11.2"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
REGISTER_OP_CUDA_KERNEL(
    sparse_attention,
    ops::SparseAttentionCUDAKernel<plf::CUDADeviceContext, float>,
    ops::SparseAttentionCUDAKernel<plf::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    sparse_attention_grad,
    ops::SparseAttentionGradCUDAKernel<plf::CUDADeviceContext, float>,
    ops::SparseAttentionGradCUDAKernel<plf::CUDADeviceContext, double>);
