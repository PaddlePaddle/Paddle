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

template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask, T val,
                                                int width = warpSize) {
  return __shfl_xor_sync(mask, val, width);
}

template <typename T, int batch_size, int warp_size>
__device__ __forceinline__ void WarpReduceSum(T* sum) {
#pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < batch_size; ++i) {
      T sum_val = CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int batch_size, int warp_size>
__device__ __forceinline__ void WarpReduceMax(T* sum) {
#pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < batch_size; ++i) {
      T max_val = CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, int BlockSize, int BlockNnzMax>
__global__ void BlockSparseSoftmaxForward(T* softmax, const T* src, T scale,
                                          const T* kp_mask, const T* attn_mask,
                                          const int* layout_rowptr,
                                          const int* layout_colindex,
                                          int num_rows) {
  // current thread related info
  const int WarpSize = 32;
  const int cur_row = blockIdx.x * blockDim.y + threadIdx.y;
  if (cur_row < num_rows) {
    const int cur_block_row = cur_row / BlockSize;
    const int cur_block_nnz =
        layout_rowptr[cur_block_row + 1] - layout_rowptr[cur_block_row];

    T srcdata[(BlockSize * BlockNnzMax + WarpSize - 1) / WarpSize] = {0};
    T attndata[(BlockSize * BlockNnzMax + WarpSize - 1) / WarpSize] = {0};

    // read tensor data, attn mask
    const int iter = (cur_block_nnz + WarpSize - 1) / WarpSize;
    const T* srcptr = src + layout_rowptr[cur_block_row];

    const T* attnptr = (attn_mask == nullptr)
                           ? nullptr
                           : (attn_mask + cur_block_row * num_rows);
    // the coloumn start index in current row
    const int* colindex = layout_colindex + layout_rowptr[cur_block_row];
    for (int j = 0; j < iter; j++) {
      int cur_block_col = j * WarpSize + threadIdx.x;
      int cur_reg_index = j;
      if (cur_block_col < cur_block_nnz) {
        // read kp mask
        T cur_kp_mask;
        if ((kp_mask != nullptr) &&
            std::abs(kp_mask[colindex[cur_block_col]]) <
                std::numeric_limits<T>::epsilon()) {
          cur_kp_mask = -std::numeric_limits<T>::infinity();
        } else {
          cur_kp_mask = 0;
        }
        // do mask operation
        if ((attnptr != nullptr) &&
            std::abs(attnptr[colindex[cur_block_col]]) <
                std::numeric_limits<T>::epsilon()) {
          srcdata[cur_reg_index] =
              -std::numeric_limits<T>::infinity() * scale + cur_kp_mask;
        } else {
          srcdata[cur_reg_index] = scale * srcptr[cur_block_col] + cur_kp_mask;
        }
      } else {
        srcdata[cur_reg_index] = -std::numeric_limits<T>::infinity();
      }
    }

    // max value
    T max_value = srcdata[0];
    const int kIteration =
        (cur_block_nnz * BlockSize + WarpSize - 1) / WarpSize;
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
    T* softmaxptr = softmax + layout_rowptr[cur_block_row];
    for (int j = 0; j < iter; j++) {
      int cur_block_col = j * WarpSize + threadIdx.x;
      int cur_reg_index = j;
      if (cur_block_col < cur_block_nnz) {
        softmaxptr[cur_block_col] = srcdata[cur_reg_index] / sum;
      }
    }
  }
}

template <typename T, int BlockSize, int BlockNnzMax>
__global__ void BlockSparseSoftmaxBackward(T* dst, const T* grad, const T* src,
                                           T scale, const int* layout_rowptr,
                                           const int* layout_colindex,
                                           int num_rows) {
  // current thread related info
  const int WarpSize = 32;
  const int cur_row = blockIdx.x * blockDim.y + threadIdx.y;
  if (cur_row < num_rows) {
    const int cur_block_row = cur_row / BlockSize;
    const int cur_block_nnz =
        layout_rowptr[cur_block_row + 1] - layout_rowptr[cur_block_row];

    T srcdata[(BlockSize * BlockNnzMax + WarpSize - 1) / WarpSize];
    T graddata[(BlockSize * BlockNnzMax + WarpSize - 1) / WarpSize];

    // read tensor data, attn mask
    const int iter = (cur_block_nnz + WarpSize - 1) / WarpSize;
    const T* srcptr = src + layout_rowptr[cur_block_row];
    const T* gradptr = grad + layout_rowptr[cur_block_row];
    for (int j = 0; j < iter; j++) {
      int cur_block_col = j * WarpSize + threadIdx.x;
      int cur_reg_index = j;
      if (cur_block_col < cur_block_nnz) {
        srcdata[cur_reg_index] = srcptr[cur_block_col];
        graddata[cur_reg_index] = gradptr[cur_block_col];
      } else {
        srcdata[cur_reg_index] = 0;
        graddata[cur_reg_index] = 0;
      }
    }

    T sum = 0;
    const int kIteration =
        (cur_block_nnz * BlockSize + WarpSize - 1) / WarpSize;
#pragma unroll
    for (int it = 0; it < kIteration; ++it) {
      sum += srcdata[it] * graddata[it];
    }
    WarpReduceSum<T, 1, WarpSize>(&sum);

    // compute softmax and write out
    T* dstptr = dst + layout_rowptr[cur_block_row];
    for (int j = 0; j < iter; j++) {
      int cur_block_col = j * WarpSize + threadIdx.x;
      int cur_reg_index = j;
      if (cur_block_col < cur_block_nnz) {
        dstptr[cur_block_col] =
            scale * srcdata[cur_reg_index] * (graddata[cur_reg_index] - sum);
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
                          const int num_rows, const int num_cols,
                          const Tensor* key_padding_mask,
                          const Tensor* attn_mask) {
  const int* offset_data = offset->data<int>();
  const int* columns_data = columns->data<int>();
  T* input_data = input->data<T>();
  T* output_data = output->data<T>();
  // Add mask
  const T* key_padding_mask_data =
      (key_padding_mask != nullptr) ? key_padding_mask->data<T>() : nullptr;
  const T* attn_mask_data =
      (attn_mask != nullptr) ? attn_mask->data<T>() : nullptr;

  const int block_size = 1;
  dim3 blocks(32, 4, 1);
  int grid = (num_rows * block_size + 3) / 4;
  T scaling = static_cast<T>(1.0) / sqrt(static_cast<T>(num_cols));

  if (num_cols <= 4) {
    BlockSparseSoftmaxForward<T, block_size, 4><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 4 && num_cols <= 8) {
    BlockSparseSoftmaxForward<T, block_size, 8><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 8 && num_cols <= 16) {
    BlockSparseSoftmaxForward<T, block_size, 16><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 16 && num_cols <= 32) {
    BlockSparseSoftmaxForward<T, block_size, 32><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 32 && num_cols <= 64) {
    BlockSparseSoftmaxForward<T, block_size, 64><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 64 && num_cols <= 128) {
    BlockSparseSoftmaxForward<T, block_size, 128><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 128 && num_cols <= 256) {
    BlockSparseSoftmaxForward<T, block_size, 256><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else if (num_cols > 256 && num_cols <= 512) {
    BlockSparseSoftmaxForward<T, block_size, 512><<<grid, blocks>>>(
        output_data, input_data, scaling, key_padding_mask_data, attn_mask_data,
        offset_data, columns_data, num_rows);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The head_dim of query in sparse_attention op should less or equal "
        "512"));
  }
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

  const int block_size = 1;
  dim3 blocks(32, 4, 1);
  int grid = (num_rows * block_size + 3) / 4;
  T scaling = static_cast<T>(1.0) / sqrt(static_cast<T>(num_cols));

  if (num_cols <= 4) {
    BlockSparseSoftmaxBackward<T, block_size, 4><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 4 && num_cols <= 8) {
    BlockSparseSoftmaxBackward<T, block_size, 8><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 8 && num_cols <= 16) {
    BlockSparseSoftmaxBackward<T, block_size, 16><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 16 && num_cols <= 32) {
    BlockSparseSoftmaxBackward<T, block_size, 32><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 32 && num_cols <= 64) {
    BlockSparseSoftmaxBackward<T, block_size, 64><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 64 && num_cols <= 128) {
    BlockSparseSoftmaxBackward<T, block_size, 128><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 128 && num_cols <= 256) {
    BlockSparseSoftmaxBackward<T, block_size, 256><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else if (num_cols > 256 && num_cols <= 512) {
    BlockSparseSoftmaxBackward<T, block_size, 512><<<grid, blocks>>>(
        dx_data, dout_data, out_data, scaling, offset_data, columns_data,
        num_rows);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The head_dim of query in sparse_attention op should less or equal "
        "512"));
  }
}

using VarType = framework::proto::VarType;
inline cudaDataType_t GetGpuType(const VarType::Type data_type) {
  if (data_type == VarType::FP32) {
    return CUDA_R_32F;
  } else if (data_type == VarType::FP64) {
    return CUDA_R_64F;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Not support tensor type in sparse_attention OP: %s",
        framework::DataTypeToString(data_type)));
  }
}

inline cusparseOperation_t GetTransposeOperation(const bool transpose) {
  if (transpose) {
    return CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
}

void CusparseDestroy(cusparseDnMatDescr_t* dn_mat_first,
                     cusparseDnMatDescr_t* dn_mat_second,
                     cusparseSpMatDescr_t* sp_mat) {
  platform::dynload::cusparseDestroyDnMat(*dn_mat_first);
  platform::dynload::cusparseDestroyDnMat(*dn_mat_second);
  platform::dynload::cusparseDestroySpMat(*sp_mat);
}

/*
input: dense A (num_rows,num_cols), dense B (num_rows,num_cols)
output: sparse C in CSR format (num_rows,num_rows)
*/
template <typename DeviceContext, typename T>
void DotSdd(const platform::CUDADeviceContext& ctx, const Tensor* a,
            const Tensor* b, const Tensor* c_offset, const Tensor* c_columns,
            Tensor* c_value, const int num_rows, const int num_cols,
            const bool a_transpose, const bool b_transpose) {
  const T* a_data = a->data<T>();
  const T* b_data = b->data<T>();
  const int* c_offset_data = c_offset->data<int>();
  const int* c_columns_data = c_columns->data<int>();
  T* c_value_data = c_value->data<T>();

  cudaDataType_t gpu_type =
      GetGpuType(framework::TransToProtoVarType(c_value->dtype()));
  cusparseHandle_t handle = nullptr;
  cusparseDnMatDescr_t mat_a, mat_b;
  cusparseSpMatDescr_t mat_c;
  platform::dynload::cusparseCreate(&handle);

  // Create dense matrix A
  platform::dynload::cusparseCreateDnMat(&mat_a, num_rows, num_cols, num_cols,
                                         const_cast<T*>(a_data), gpu_type,
                                         CUSPARSE_ORDER_ROW);
  // Create dense matrix B
  platform::dynload::cusparseCreateDnMat(&mat_b, num_rows, num_cols, num_cols,
                                         const_cast<T*>(b_data), gpu_type,
                                         CUSPARSE_ORDER_ROW);
  // Create sparse matrix C in CSR format
  int c_nnz = c_columns->dims()[1];
  platform::dynload::cusparseCreateCsr(
      &mat_c, num_rows, num_rows, c_nnz, const_cast<int*>(c_offset_data),
      const_cast<int*>(c_columns_data), c_value_data, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, gpu_type);

  T alpha = 1;
  T beta = 0;

  size_t buffer_size = 0;
  platform::dynload::cusparseSDDMM_bufferSize(
      handle, GetTransposeOperation(a_transpose),
      GetTransposeOperation(b_transpose), &alpha, mat_a, mat_b, &beta, mat_c,
      gpu_type, CUSPARSE_SDDMM_ALG_DEFAULT, &buffer_size);
  auto d_buffer_ptr = paddle::memory::Alloc(ctx, buffer_size);
  void* d_buffer = static_cast<void*>(d_buffer_ptr->ptr());

  platform::dynload::cusparseSDDMM(handle, GetTransposeOperation(a_transpose),
                                   GetTransposeOperation(b_transpose), &alpha,
                                   mat_a, mat_b, &beta, mat_c, gpu_type,
                                   CUSPARSE_SDDMM_ALG_DEFAULT, d_buffer);

  CusparseDestroy(&mat_a, &mat_b, &mat_c);
  platform::dynload::cusparseDestroy(handle);
}

/*
input: sparse A in CSR format (num_rows,num_rows), dense B (num_rows,num_cols)
output: dense C (num_rows,num_cols)
*/
template <typename DeviceContext, typename T>
void DotDsd(const platform::CUDADeviceContext& ctx, const Tensor* a_offset,
            const Tensor* a_columns, const Tensor* a_value, const Tensor* b,
            Tensor* c, const int num_rows, const int num_cols,
            const bool a_transpose, const bool b_transpose) {
  const int* a_offset_data = a_offset->data<int>();
  const int* a_columns_data = a_columns->data<int>();
  const T* a_value_data = a_value->data<T>();
  const T* b_data = b->data<T>();
  T* c_data = c->data<T>();

  cudaDataType_t gpu_type =
      GetGpuType(framework::TransToProtoVarType(c->dtype()));
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t mat_a;
  cusparseDnMatDescr_t mat_b, mat_c;
  platform::dynload::cusparseCreate(&handle);

  // Create sparse matrix A in CSR format
  int a_nnz = a_columns->dims()[1];
  platform::dynload::cusparseCreateCsr(
      &mat_a, num_rows, num_rows, a_nnz, const_cast<int*>(a_offset_data),
      const_cast<int*>(a_columns_data), const_cast<T*>(a_value_data),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      gpu_type);

  // Create dense matrix B
  platform::dynload::cusparseCreateDnMat(&mat_b, num_rows, num_cols, num_cols,
                                         const_cast<T*>(b_data), gpu_type,
                                         CUSPARSE_ORDER_ROW);
  // Create dense matrix C
  platform::dynload::cusparseCreateDnMat(&mat_c, num_rows, num_cols, num_cols,
                                         c_data, gpu_type, CUSPARSE_ORDER_ROW);

  T alpha = 1;
  T beta = 0;

  size_t buffer_size = 0;
  // allocate an external buffer if needed
  platform::dynload::cusparseSpMM_bufferSize(
      handle, GetTransposeOperation(a_transpose),
      GetTransposeOperation(b_transpose), &alpha, mat_a, mat_b, &beta, mat_c,
      gpu_type, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
  auto d_buffer_ptr = paddle::memory::Alloc(ctx, buffer_size);
  void* d_buffer = static_cast<void*>(d_buffer_ptr->ptr());

  platform::dynload::cusparseSpMM(handle, GetTransposeOperation(a_transpose),
                                  GetTransposeOperation(b_transpose), &alpha,
                                  mat_a, mat_b, &beta, mat_c, gpu_type,
                                  CUSPARSE_SPMM_ALG_DEFAULT, d_buffer);

  CusparseDestroy(&mat_b, &mat_c, &mat_a);
  platform::dynload::cusparseDestroy(handle);
}

std::vector<Tensor> GetSplitTensor(Tensor* input) {
  auto dims = input->dims();
  int batch_size = dims[0];
  int num_heads = dims[1];
  std::vector<int> new_dims(dims.size() - 1);
  new_dims[0] = batch_size * num_heads;
  for (int i = 1; i < new_dims.size(); i++) {
    new_dims[i] = dims[i + 1];
  }
  input->Resize(phi::make_ddim(new_dims));
  return input->Split(1, 0);
}

template <typename DeviceContext, typename T>
class SparseAttentionCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto query = *ctx.Input<Tensor>("Q");
    auto key = *ctx.Input<Tensor>("K");
    auto value = *ctx.Input<Tensor>("V");
    auto offset = *ctx.Input<Tensor>("Offset");
    auto columns = *ctx.Input<Tensor>("Columns");
    auto output_ptr = ctx.Output<Tensor>("Out");
    output_ptr->mutable_data<T>(ctx.GetPlace());
    auto sparse_dot_sdd_ptr = ctx.Output<Tensor>("SparseDotSdd");
    sparse_dot_sdd_ptr->mutable_data<T>(ctx.GetPlace());
    auto softmax_ptr = ctx.Output<Tensor>("Softmax");
    softmax_ptr->mutable_data<T>(ctx.GetPlace());
    // add Mask
    auto* key_padding_mask = ctx.HasInput("KeyPaddingMask")
                                 ? ctx.Input<Tensor>("KeyPaddingMask")
                                 : nullptr;
    auto* attn_mask =
        ctx.HasInput("AttnMask") ? ctx.Input<Tensor>("AttnMask") : nullptr;

    auto output = *output_ptr;
    auto result_sdd = *sparse_dot_sdd_ptr;
    auto result_softmax = *softmax_ptr;

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

      if (key_padding_mask != nullptr && attn_mask != nullptr) {
        SparseSoftmaxForward<DeviceContext, T>(
            dev_ctx, &offset_lists[i], &columns_lists[i], &result_sdd_lists[i],
            &result_softmax_lists[i], 1, M, N,
            key_padding_mask + (i / num_heads) * M, attn_mask);
      } else if (key_padding_mask != nullptr && attn_mask == nullptr) {
        SparseSoftmaxForward<DeviceContext, T>(
            dev_ctx, &offset_lists[i], &columns_lists[i], &result_sdd_lists[i],
            &result_softmax_lists[i], 1, M, N,
            key_padding_mask + (i / num_heads) * M, nullptr);
      } else if (key_padding_mask == nullptr && attn_mask != nullptr) {
        SparseSoftmaxForward<DeviceContext, T>(
            dev_ctx, &offset_lists[i], &columns_lists[i], &result_sdd_lists[i],
            &result_softmax_lists[i], 1, M, N, nullptr, attn_mask);
      } else {
        SparseSoftmaxForward<DeviceContext, T>(
            dev_ctx, &offset_lists[i], &columns_lists[i], &result_sdd_lists[i],
            &result_softmax_lists[i], 1, M, N, nullptr, nullptr);
      }

      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &result_softmax_lists[i], &value_lists[i],
                               &output_lists[i], M, N, false, false);
    }
  }
};

template <typename DeviceContext, typename T>
class SparseAttentionGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto query = *ctx.Input<Tensor>("Q");
    auto key = *ctx.Input<Tensor>("K");
    auto value = *ctx.Input<Tensor>("V");
    auto offset = *ctx.Input<Tensor>("Offset");
    auto columns = *ctx.Input<Tensor>("Columns");
    auto sparse_dot_sdd = *ctx.Input<Tensor>("SparseDotSdd");
    auto softmax = *ctx.Input<Tensor>("Softmax");
    auto dout = *ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dquery_ptr = ctx.Output<Tensor>(framework::GradVarName("Q"));
    auto* dkey_ptr = ctx.Output<Tensor>(framework::GradVarName("K"));
    auto* dvalue_ptr = ctx.Output<Tensor>(framework::GradVarName("V"));
    dquery_ptr->mutable_data<T>(ctx.GetPlace());
    dkey_ptr->mutable_data<T>(ctx.GetPlace());
    dvalue_ptr->mutable_data<T>(ctx.GetPlace());
    auto dquery = *dquery_ptr;
    auto dkey = *dkey_ptr;
    auto dvalue = *dvalue_ptr;

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
    std::vector<Tensor> sparse_dot_sdd_lists = GetSplitTensor(&sparse_dot_sdd);
    std::vector<Tensor> softmax_lists = GetSplitTensor(&softmax);
    std::vector<Tensor> dout_lists = GetSplitTensor(&dout);
    std::vector<Tensor> dquery_lists = GetSplitTensor(&dquery);
    std::vector<Tensor> dkey_lists = GetSplitTensor(&dkey);
    std::vector<Tensor> dvalue_lists = GetSplitTensor(&dvalue);

    const int iter_num = batch_size * num_heads;
    const auto& dev_ctx = ctx.cuda_device_context();
    for (int i = 0; i < iter_num; i++) {
      // dValue = transpose(result_softmax) * dOut
      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &softmax_lists[i], &dout_lists[i],
                               &dvalue_lists[i], M, N, true, false);

      // dSoftmax = dOut * transpose(Value)
      int nnz_num = columns.dims()[0];
      Tensor dsoftmax;
      dsoftmax.Resize({nnz_num});
      dsoftmax.mutable_data<T>(ctx.GetPlace());
      DotSdd<DeviceContext, T>(dev_ctx, &dout_lists[i], &value_lists[i],
                               &offset_lists[i], &columns_lists[i], &dsoftmax,
                               M, N, false, true);

      // dSparseDotSdd = dSoftmax * softmax'(SparseDotSdd)
      Tensor dsparse_dot_sdd;
      dsparse_dot_sdd.Resize({nnz_num});
      dsparse_dot_sdd.mutable_data<T>(ctx.GetPlace());
      SparseSoftmaxBackward<DeviceContext, T>(
          dev_ctx, &offset_lists[i], &columns_lists[i], &dsparse_dot_sdd,
          &dsoftmax, &softmax_lists[i], 1, M, N);

      // dQuery = dSparseDotSdd * Key
      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &dsparse_dot_sdd, &key_lists[i],
                               &dquery_lists[i], M, N, false, false);

      // dKey = transpose(dSparseDotSdd) * Query
      DotDsd<DeviceContext, T>(dev_ctx, &offset_lists[i], &columns_lists[i],
                               &dsparse_dot_sdd, &query_lists[i],
                               &dkey_lists[i], M, N, true, false);
    }
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
