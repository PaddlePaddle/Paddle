// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>

#include <limits>
#include <string>
#include <vector>
#include "paddle/phi/common/memory_utils.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cusparse.h"
#endif
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/utils/optional.h"

namespace phi {
#if defined(PADDLE_WITH_CUDA)
template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask,
                                                T val,
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
__global__ void BlockSparseSoftmaxForward(T* softmax,
                                          const T* src,
                                          T scale,
                                          const T* kp_mask,
                                          const T* attn_mask,
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
    // the column start index in current row
    const int* colindex = layout_colindex + layout_rowptr[cur_block_row];
    for (int j = 0; j < iter; j++) {
      int cur_block_col = j * WarpSize + threadIdx.x;
      int cur_reg_index = j;
      if (cur_block_col < cur_block_nnz) {
        // read kp mask
        T cur_kp_mask;
        if ((kp_mask != nullptr) && std::abs(kp_mask[colindex[cur_block_col]]) <
                                        std::numeric_limits<T>::epsilon()) {
          cur_kp_mask = -std::numeric_limits<T>::infinity();
        } else {
          cur_kp_mask = 0;
        }
        // do mask operation
        if ((attnptr != nullptr) && std::abs(attnptr[colindex[cur_block_col]]) <
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
__global__ void BlockSparseSoftmaxBackward(T* dst,
                                           const T* grad,
                                           const T* src,
                                           T scale,
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

/*
input: sparse C in CSR format (num_rows,num_rows)
output: sparse C after softmax operation
*/
template <typename DeviceContext, typename T>
void SparseSoftmaxForward(const phi::GPUContext& ctx,
                          const phi::DenseTensor* offset,
                          const phi::DenseTensor* columns,
                          phi::DenseTensor* input,
                          phi::DenseTensor* output,
                          const int blocksize,
                          const int num_rows,
                          const int num_cols,
                          const phi::DenseTensor* key_padding_mask,
                          const phi::DenseTensor* attn_mask) {
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
    BlockSparseSoftmaxForward<T, block_size, 4>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 4 && num_cols <= 8) {
    BlockSparseSoftmaxForward<T, block_size, 8>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 8 && num_cols <= 16) {
    BlockSparseSoftmaxForward<T, block_size, 16>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 16 && num_cols <= 32) {
    BlockSparseSoftmaxForward<T, block_size, 32>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 32 && num_cols <= 64) {
    BlockSparseSoftmaxForward<T, block_size, 64>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 64 && num_cols <= 128) {
    BlockSparseSoftmaxForward<T, block_size, 128>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 128 && num_cols <= 256) {
    BlockSparseSoftmaxForward<T, block_size, 256>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 256 && num_cols <= 512) {
    BlockSparseSoftmaxForward<T, block_size, 512>
        <<<grid, blocks>>>(output_data,
                           input_data,
                           scaling,
                           key_padding_mask_data,
                           attn_mask_data,
                           offset_data,
                           columns_data,
                           num_rows);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The head_dim of query in sparse_attention op should less or equal "
        "512"));
  }
}

template <typename DeviceContext, typename T>
void SparseSoftmaxBackward(const phi::GPUContext& ctx,
                           const phi::DenseTensor* offset,
                           const phi::DenseTensor* columns,
                           phi::DenseTensor* dx,
                           const phi::DenseTensor* dout,
                           const phi::DenseTensor* out,
                           const int blocksize,
                           const int num_rows,
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
    BlockSparseSoftmaxBackward<T, block_size, 4><<<grid, blocks>>>(dx_data,
                                                                   dout_data,
                                                                   out_data,
                                                                   scaling,
                                                                   offset_data,
                                                                   columns_data,
                                                                   num_rows);
  } else if (num_cols > 4 && num_cols <= 8) {
    BlockSparseSoftmaxBackward<T, block_size, 8><<<grid, blocks>>>(dx_data,
                                                                   dout_data,
                                                                   out_data,
                                                                   scaling,
                                                                   offset_data,
                                                                   columns_data,
                                                                   num_rows);
  } else if (num_cols > 8 && num_cols <= 16) {
    BlockSparseSoftmaxBackward<T, block_size, 16>
        <<<grid, blocks>>>(dx_data,
                           dout_data,
                           out_data,
                           scaling,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 16 && num_cols <= 32) {
    BlockSparseSoftmaxBackward<T, block_size, 32>
        <<<grid, blocks>>>(dx_data,
                           dout_data,
                           out_data,
                           scaling,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 32 && num_cols <= 64) {
    BlockSparseSoftmaxBackward<T, block_size, 64>
        <<<grid, blocks>>>(dx_data,
                           dout_data,
                           out_data,
                           scaling,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 64 && num_cols <= 128) {
    BlockSparseSoftmaxBackward<T, block_size, 128>
        <<<grid, blocks>>>(dx_data,
                           dout_data,
                           out_data,
                           scaling,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 128 && num_cols <= 256) {
    BlockSparseSoftmaxBackward<T, block_size, 256>
        <<<grid, blocks>>>(dx_data,
                           dout_data,
                           out_data,
                           scaling,
                           offset_data,
                           columns_data,
                           num_rows);
  } else if (num_cols > 256 && num_cols <= 512) {
    BlockSparseSoftmaxBackward<T, block_size, 512>
        <<<grid, blocks>>>(dx_data,
                           dout_data,
                           out_data,
                           scaling,
                           offset_data,
                           columns_data,
                           num_rows);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The head_dim of query in sparse_attention op should less or equal "
        "512"));
  }
}

inline cudaDataType_t GetGpuType(const DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return CUDA_R_32F;
  } else if (data_type == DataType::FLOAT64) {
    return CUDA_R_64F;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Not support tensor type in sparse_attention OP: %s",
        phi::DataTypeToString(data_type)));
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
  phi::dynload::cusparseDestroyDnMat(*dn_mat_first);
  phi::dynload::cusparseDestroyDnMat(*dn_mat_second);
  phi::dynload::cusparseDestroySpMat(*sp_mat);
}
#endif

/*
input: dense A (num_rows,num_cols), dense B (num_rows,num_cols)
output: sparse C in CSR format (num_rows,num_rows)
*/
template <typename DeviceContext, typename T>
void DotSdd(const phi::GPUContext& ctx,
            const phi::DenseTensor* a,
            const phi::DenseTensor* b,
            const phi::DenseTensor* c_offset,
            const phi::DenseTensor* c_columns,
            phi::DenseTensor* c_value,
            const int num_rows,
            const int num_cols,
            const bool a_transpose,
            const bool b_transpose) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11030
  const T* a_data = a->data<T>();
  const T* b_data = b->data<T>();
  const int* c_offset_data = c_offset->data<int>();
  const int* c_columns_data = c_columns->data<int>();
  T* c_value_data = c_value->data<T>();

  cudaDataType_t gpu_type = GetGpuType(c_value->dtype());
  cusparseHandle_t handle = nullptr;
  cusparseDnMatDescr_t mat_a, mat_b;
  cusparseSpMatDescr_t mat_c;
  phi::dynload::cusparseCreate(&handle);

  // Create dense matrix A
  phi::dynload::cusparseCreateDnMat(&mat_a,
                                    num_rows,
                                    num_cols,
                                    num_cols,
                                    const_cast<T*>(a_data),
                                    gpu_type,
                                    CUSPARSE_ORDER_ROW);
  // Create dense matrix B
  phi::dynload::cusparseCreateDnMat(&mat_b,
                                    num_rows,
                                    num_cols,
                                    num_cols,
                                    const_cast<T*>(b_data),
                                    gpu_type,
                                    CUSPARSE_ORDER_ROW);
  // Create sparse matrix C in CSR format
  int c_nnz = c_columns->numel();
  phi::dynload::cusparseCreateCsr(&mat_c,
                                  num_rows,
                                  num_rows,
                                  c_nnz,
                                  const_cast<int*>(c_offset_data),
                                  const_cast<int*>(c_columns_data),
                                  c_value_data,
                                  CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO,
                                  gpu_type);

  T alpha = 1;
  T beta = 0;

  size_t buffer_size = 0;
  phi::dynload::cusparseSDDMM_bufferSize(handle,
                                         GetTransposeOperation(a_transpose),
                                         GetTransposeOperation(b_transpose),
                                         &alpha,
                                         mat_a,
                                         mat_b,
                                         &beta,
                                         mat_c,
                                         gpu_type,
                                         CUSPARSE_SDDMM_ALG_DEFAULT,
                                         &buffer_size);
  auto d_buffer_ptr = phi::memory_utils::Alloc(
      ctx.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  void* d_buffer = static_cast<void*>(d_buffer_ptr->ptr());

  phi::dynload::cusparseSDDMM(handle,
                              GetTransposeOperation(a_transpose),
                              GetTransposeOperation(b_transpose),
                              &alpha,
                              mat_a,
                              mat_b,
                              &beta,
                              mat_c,
                              gpu_type,
                              CUSPARSE_SDDMM_ALG_DEFAULT,
                              d_buffer);

  CusparseDestroy(&mat_a, &mat_b, &mat_c);
  phi::dynload::cusparseDestroy(handle);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "DotSdd use cusparseSDDMM, which is supported "
      "from CUDA 11.3"));
#endif
}

/*
input: sparse A in CSR format (num_rows,num_rows), dense B (num_rows,num_cols)
output: dense C (num_rows,num_cols)
*/
template <typename DeviceContext, typename T>
void DotDsd(const phi::GPUContext& ctx,
            const phi::DenseTensor* a_offset,
            const phi::DenseTensor* a_columns,
            const phi::DenseTensor* a_value,
            const phi::DenseTensor* b,
            phi::DenseTensor* c,
            const int num_rows,
            const int num_cols,
            const bool a_transpose,
            const bool b_transpose) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11000
  const int* a_offset_data = a_offset->data<int>();
  const int* a_columns_data = a_columns->data<int>();
  const T* a_value_data = a_value->data<T>();
  const T* b_data = b->data<T>();
  T* c_data = c->data<T>();

  cudaDataType_t gpu_type = GetGpuType(c->dtype());
  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t mat_a;
  cusparseDnMatDescr_t mat_b, mat_c;
  phi::dynload::cusparseCreate(&handle);

  // Create sparse matrix A in CSR format
  int a_nnz = a_columns->numel();
  phi::dynload::cusparseCreateCsr(&mat_a,
                                  num_rows,
                                  num_rows,
                                  a_nnz,
                                  const_cast<int*>(a_offset_data),
                                  const_cast<int*>(a_columns_data),
                                  const_cast<T*>(a_value_data),
                                  CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO,
                                  gpu_type);

  // Create dense matrix B
  phi::dynload::cusparseCreateDnMat(&mat_b,
                                    num_rows,
                                    num_cols,
                                    num_cols,
                                    const_cast<T*>(b_data),
                                    gpu_type,
                                    CUSPARSE_ORDER_ROW);
  // Create dense matrix C
  phi::dynload::cusparseCreateDnMat(&mat_c,
                                    num_rows,
                                    num_cols,
                                    num_cols,
                                    c_data,
                                    gpu_type,
                                    CUSPARSE_ORDER_ROW);

  T alpha = 1;
  T beta = 0;

  size_t buffer_size = 0;
  // allocate an external buffer if needed
  phi::dynload::cusparseSpMM_bufferSize(handle,
                                        GetTransposeOperation(a_transpose),
                                        GetTransposeOperation(b_transpose),
                                        &alpha,
                                        mat_a,
                                        mat_b,
                                        &beta,
                                        mat_c,
                                        gpu_type,
                                        CUSPARSE_SPMM_ALG_DEFAULT,
                                        &buffer_size);
  auto d_buffer_ptr = phi::memory_utils::Alloc(
      ctx.GetPlace(),
      buffer_size,
      phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
  void* d_buffer = static_cast<void*>(d_buffer_ptr->ptr());

  phi::dynload::cusparseSpMM(handle,
                             GetTransposeOperation(a_transpose),
                             GetTransposeOperation(b_transpose),
                             &alpha,
                             mat_a,
                             mat_b,
                             &beta,
                             mat_c,
                             gpu_type,
                             CUSPARSE_SPMM_ALG_DEFAULT,
                             d_buffer);

  CusparseDestroy(&mat_b, &mat_c, &mat_a);
  phi::dynload::cusparseDestroy(handle);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "DotDsd use cusparseSpMM, which is supported "
      "from CUDA 11.0"));
#endif
}

std::vector<phi::DenseTensor> GetSplitTensor(phi::DenseTensor* input) {
  auto dims = input->dims();
  int batch_size = dims[0];
  int num_heads = dims[1];
  std::vector<int> new_dims(dims.size() - 1);
  new_dims[0] = batch_size * num_heads;
  for (int i = 1; i < new_dims.size(); i++) {
    new_dims[i] = dims[i + 1];
  }
  input->Resize(common::make_ddim(new_dims));
  return input->Split(1, 0);
}

template <typename T, typename Context>
void SparseAttentionCUDAKernel(
    const Context& dev_ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& offset,
    const DenseTensor& columns,
    const paddle::optional<DenseTensor>& key_padding_mask,
    const paddle::optional<DenseTensor>& attn_mask,
    DenseTensor* out,
    DenseTensor* sparse_dot_sdd,
    DenseTensor* softmax) {
#if defined(PADDLE_WITH_CUDA)
  auto query = q;
  auto key = k;
  auto value = v;
  auto output_ptr = out;
  dev_ctx.template Alloc<T>(out);
  auto sparse_dot_sdd_ptr = sparse_dot_sdd;
  dev_ctx.template Alloc<T>(sparse_dot_sdd);
  auto softmax_ptr = softmax;
  dev_ctx.template Alloc<T>(softmax);

  auto output = *output_ptr;
  auto result_sdd = *sparse_dot_sdd_ptr;
  auto result_softmax = *softmax_ptr;

  auto query_dims = query.dims();
  int batch_size = query_dims[0];
  int num_heads = query_dims[1];
  int M = query_dims[2];
  int N = query_dims[3];

  DenseTensor q2 = q;
  DenseTensor k2 = k;
  DenseTensor v2 = v;
  DenseTensor offset2 = offset;
  DenseTensor columns2 = columns;
  std::vector<phi::DenseTensor> query_lists = GetSplitTensor(&q2);
  std::vector<phi::DenseTensor> key_lists = GetSplitTensor(&k2);
  std::vector<phi::DenseTensor> value_lists = GetSplitTensor(&v2);
  std::vector<phi::DenseTensor> offset_lists = GetSplitTensor(&offset2);
  std::vector<phi::DenseTensor> columns_lists = GetSplitTensor(&columns2);
  std::vector<phi::DenseTensor> result_sdd_lists = GetSplitTensor(&result_sdd);
  std::vector<phi::DenseTensor> result_softmax_lists =
      GetSplitTensor(&result_softmax);
  std::vector<phi::DenseTensor> output_lists = GetSplitTensor(&output);

  const int iter_num = batch_size * num_heads;
  for (int i = 0; i < iter_num; i++) {
    DotSdd<Context, T>(dev_ctx,
                       &query_lists[i],
                       &key_lists[i],
                       &offset_lists[i],
                       &columns_lists[i],
                       &result_sdd_lists[i],
                       M,
                       N,
                       false,
                       true);

    if (key_padding_mask && attn_mask) {
      SparseSoftmaxForward<Context, T>(
          dev_ctx,
          &offset_lists[i],
          &columns_lists[i],
          &result_sdd_lists[i],
          &result_softmax_lists[i],
          1,
          M,
          N,
          key_padding_mask.get_ptr() + (i / num_heads) * M,
          attn_mask.get_ptr());
    } else if (key_padding_mask && !attn_mask.is_initialized()) {
      SparseSoftmaxForward<Context, T>(
          dev_ctx,
          &offset_lists[i],
          &columns_lists[i],
          &result_sdd_lists[i],
          &result_softmax_lists[i],
          1,
          M,
          N,
          key_padding_mask.get_ptr() + (i / num_heads) * M,
          nullptr);
    } else if (!key_padding_mask.is_initialized() && attn_mask) {
      SparseSoftmaxForward<Context, T>(dev_ctx,
                                       &offset_lists[i],
                                       &columns_lists[i],
                                       &result_sdd_lists[i],
                                       &result_softmax_lists[i],
                                       1,
                                       M,
                                       N,
                                       nullptr,
                                       attn_mask.get_ptr());
    } else {
      SparseSoftmaxForward<Context, T>(dev_ctx,
                                       &offset_lists[i],
                                       &columns_lists[i],
                                       &result_sdd_lists[i],
                                       &result_softmax_lists[i],
                                       1,
                                       M,
                                       N,
                                       nullptr,
                                       nullptr);
    }

    DotDsd<Context, T>(dev_ctx,
                       &offset_lists[i],
                       &columns_lists[i],
                       &result_softmax_lists[i],
                       &value_lists[i],
                       &output_lists[i],
                       M,
                       N,
                       false,
                       false);
  }
#endif
}

template <typename T, typename Context>
void SparseAttentionGradCUDAKernel(const Context& dev_ctx,
                                   const DenseTensor& q,
                                   const DenseTensor& k,
                                   const DenseTensor& v,
                                   const DenseTensor& offset,
                                   const DenseTensor& columns,
                                   const DenseTensor& sparse_dot_sdd,
                                   const DenseTensor& softmax,
                                   const DenseTensor& out_grad,
                                   DenseTensor* q_grad,
                                   DenseTensor* k_grad,
                                   DenseTensor* v_grad) {
#if defined(PADDLE_WITH_CUDA)
  auto query = q;
  auto key = k;
  auto value = v;

  auto dout = out_grad;
  auto* dquery_ptr = q_grad;
  auto* dkey_ptr = k_grad;
  auto* dvalue_ptr = v_grad;
  dev_ctx.template Alloc<T>(q_grad);
  dev_ctx.template Alloc<T>(k_grad);
  dev_ctx.template Alloc<T>(v_grad);

  auto dquery = *dquery_ptr;
  auto dkey = *dkey_ptr;
  auto dvalue = *dvalue_ptr;

  auto query_dims = query.dims();
  int batch_size = query_dims[0];
  int num_heads = query_dims[1];
  int M = query_dims[2];
  int N = query_dims[3];

  DenseTensor q2 = q;
  DenseTensor k2 = k;
  DenseTensor v2 = v;
  DenseTensor offset2 = offset;
  DenseTensor columns2 = columns;
  DenseTensor sparse_dot_sdd2 = sparse_dot_sdd;
  DenseTensor softmax2 = softmax;
  DenseTensor dout2 = out_grad;
  std::vector<phi::DenseTensor> query_lists = GetSplitTensor(&q2);
  std::vector<phi::DenseTensor> key_lists = GetSplitTensor(&k2);
  std::vector<phi::DenseTensor> value_lists = GetSplitTensor(&v2);
  std::vector<phi::DenseTensor> offset_lists = GetSplitTensor(&offset2);
  std::vector<phi::DenseTensor> columns_lists = GetSplitTensor(&columns2);
  std::vector<phi::DenseTensor> sparse_dot_sdd_lists =
      GetSplitTensor(&sparse_dot_sdd2);
  std::vector<phi::DenseTensor> softmax_lists = GetSplitTensor(&softmax2);
  std::vector<phi::DenseTensor> dout_lists = GetSplitTensor(&dout2);
  std::vector<phi::DenseTensor> dquery_lists = GetSplitTensor(&dquery);
  std::vector<phi::DenseTensor> dkey_lists = GetSplitTensor(&dkey);
  std::vector<phi::DenseTensor> dvalue_lists = GetSplitTensor(&dvalue);

  const int iter_num = batch_size * num_heads;
  for (int i = 0; i < iter_num; i++) {
    // dValue = transpose(result_softmax) * dOut
    DotDsd<Context, T>(dev_ctx,
                       &offset_lists[i],
                       &columns_lists[i],
                       &softmax_lists[i],
                       &dout_lists[i],
                       &dvalue_lists[i],
                       M,
                       N,
                       true,
                       false);

    // dSoftmax = dOut * transpose(Value)
    int nnz_num = columns_lists[i].numel();
    phi::DenseTensor dsoftmax;
    dsoftmax.Resize({nnz_num});
    dev_ctx.template Alloc<T>(&dsoftmax);
    DotSdd<Context, T>(dev_ctx,
                       &dout_lists[i],
                       &value_lists[i],
                       &offset_lists[i],
                       &columns_lists[i],
                       &dsoftmax,
                       M,
                       N,
                       false,
                       true);

    // dSparseDotSdd = dSoftmax * softmax'(SparseDotSdd)
    phi::DenseTensor dsparse_dot_sdd;
    dsparse_dot_sdd.Resize({nnz_num});
    dev_ctx.template Alloc<T>(&dsparse_dot_sdd);
    SparseSoftmaxBackward<Context, T>(dev_ctx,
                                      &offset_lists[i],
                                      &columns_lists[i],
                                      &dsparse_dot_sdd,
                                      &dsoftmax,
                                      &softmax_lists[i],
                                      1,
                                      M,
                                      N);

    // dQuery = dSparseDotSdd * Key
    DotDsd<Context, T>(dev_ctx,
                       &offset_lists[i],
                       &columns_lists[i],
                       &dsparse_dot_sdd,
                       &key_lists[i],
                       &dquery_lists[i],
                       M,
                       N,
                       false,
                       false);

    // dKey = transpose(dSparseDotSdd) * Query
    DotDsd<Context, T>(dev_ctx,
                       &offset_lists[i],
                       &columns_lists[i],
                       &dsparse_dot_sdd,
                       &query_lists[i],
                       &dkey_lists[i],
                       M,
                       N,
                       true,
                       false);
  }
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(sparse_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::SparseAttentionCUDAKernel,
                   float,
                   double) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
  kernel->InputAt(4).SetDataType(phi::DataType::INT32);
}
PD_REGISTER_KERNEL(sparse_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SparseAttentionGradCUDAKernel,
                   float,
                   double) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
  kernel->InputAt(4).SetDataType(phi::DataType::INT32);
}
