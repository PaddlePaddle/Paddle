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

#include "paddle/phi/kernels/sparse/concat_kernel.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include "glog/logging.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/algorithm.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace sparse {

template <typename T>
struct PointerToPointer {
 public:
  const T* const* ins_addr{nullptr};
  __device__ inline const const T* operator[](int i) const {
    return ins_addr[i];
  }

  PointerToPointer() = default;
  PointerToPointer(const phi::GPUContext& ctx,
                   const size_t in_num,
                   const T* const* pre_alloced_host_ptr,
                   phi::Allocator::AllocationPtr* dev_ins_ptr) {
    *dev_ins_ptr = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        in_num * sizeof(T*),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    auto* restored = phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
        pre_alloced_host_ptr, in_num);
    memory_utils::Copy(ctx.GetPlace(),
                       (*dev_ins_ptr)->ptr(),
                       phi::CPUPlace(),
                       restored,
                       in_num * sizeof(T*),
                       ctx.stream());
    ins_addr = reinterpret_cast<const T* const*>((*dev_ins_ptr)->ptr());
  }
};
// 将vector的数据从host 转移到device的结构
template <typename IndexT>
struct DArray {
 public:
  IndexT* d_array{nullptr};

  __device__ inline const IndexT operator[](int i) const { return d_array[i]; }

  DArray() = default;

  DArray(const phi::GPUContext& dev_ctx,
         const std::vector<IndexT>& host_array,
         phi::Allocator::AllocationPtr* dev_col_ptr) {
    // copy offsets to device
    *dev_col_ptr = memory_utils::Alloc(
        dev_ctx.GetPlace(),
        sizeof(IndexT) * host_array.size(),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    memory_utils::Copy(dev_ctx.GetPlace(),
                       (*dev_col_ptr)->ptr(),
                       phi::CPUPlace(),
                       host_array.data(),
                       sizeof(IndexT) * host_array.size(),
                       dev_ctx.stream());
    d_array = static_cast<IndexT*>((*dev_col_ptr)->ptr());
  }

  IndexT* get_ptr() { return d_array; }
};

static void check_cat_sparse_dims(const SparseCooTensor* t,
                                  int64_t pos,
                                  DDim dims,
                                  int64_t axis,
                                  int64_t sparse_dim,
                                  int64_t dense_dim) {
  PADDLE_ENFORCE_EQ(t->sparse_dim(),
                    sparse_dim,
                    "All tensors must have the same sparse_dim ",
                    sparse_dim,
                    ", but tensor at position ",
                    pos,
                    " has ",
                    t->sparse_dim());
  PADDLE_ENFORCE_EQ(t->dense_dim(),
                    dense_dim,
                    "All tensors must have the same dense_dim ",
                    dense_dim,
                    ", but tensor at position ",
                    pos,
                    " has ",
                    t->dense_dim());
}

template <typename IndexT>
__global__ void ConcatCooSetIndicesKernel(
    const IndexT out_nnz,
    const IndexT axis,
    const IndexT in_num,
    const IndexT* __restrict__ indice_offsets,
    const IndexT* __restrict__ d_nnz_offsets,
    IndexT* out_indices) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, out_nnz, IndexT) {
    IndexT index = phi::funcs::UpperBound<IndexT, IndexT>(
        d_nnz_offsets, in_num + 1, tid_x);
    index--;

    out_indices[axis * out_nnz + tid_x] += indice_offsets[index];
  }
}

template <typename IndexT, typename PointerWrapperT, typename DarrayWrapperT>
__global__ void ConcatCsr2D0AGetHelpArrayKernel(const size_t in_num,
                                                PointerWrapperT in_crows_vec,
                                                DarrayWrapperT in_rows,
                                                IndexT* out_crows_offsets) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, in_num, IndexT) {
    out_crows_offsets[tid_x + 1] = in_crows_vec[tid_x][in_rows[tid_x + 1]];
    if (tid_x == 0) {
      out_crows_offsets[0] = 0;

      for (int i = 0; i < in_num; i++) {
        out_crows_offsets[i + 1] += out_crows_offsets[i];
      }
    }
  }
}

template <typename IndexT, typename PointerWrapperT, typename DarrayWrapperT>
__global__ void ConcatCsr2D0ASetCrowsKernel(const IndexT total_crows_offsets,
                                            const size_t in_num,
                                            PointerWrapperT in_crows_vec,
                                            DarrayWrapperT in_rows_offsets,
                                            IndexT* out_crows_offsets,
                                            IndexT* out_crows) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_crows_offsets, IndexT) {
    if (tid_x == 0) {
      out_crows[0] = 0;
    }
    // index mean the number of input  tensor
    IndexT index = phi::funcs::UpperBound<IndexT, IndexT>(
        in_rows_offsets.d_array, in_num + 1, tid_x);
    index--;
    IndexT local_col = tid_x - in_rows_offsets[index];

    out_crows[tid_x + 1] =
        in_crows_vec[index][local_col + 1] + out_crows_offsets[index];
  }
}

template <typename T,
          typename IndexT,
          typename PointerWrapperT,
          typename PointerWrapperIndexT,
          typename DarrayWrapperT>
__global__ void ConcatCsr2D0ASetValueKernel(const IndexT total_nnz,
                                            const size_t in_num,
                                            PointerWrapperT in_values,
                                            PointerWrapperIndexT in_cols,
                                            DarrayWrapperT nnz_offsets,
                                            T* out_values,
                                            IndexT* out_cols) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_nnz, IndexT) {
    IndexT index = 0;

    index = phi::funcs::UpperBound<IndexT, IndexT>(
        nnz_offsets.d_array, in_num + 1, tid_x);
    index--;
    IndexT left_nnz = tid_x - nnz_offsets[index];

    out_values[tid_x] = in_values[index][left_nnz];
    // need to add the previous tensor's col number in this line
    out_cols[tid_x] = in_cols[index][left_nnz];
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr2D1AGetHelpArrayKernel(const IndexT total_rows,
                                                const size_t in_num,
                                                const IndexT rows,
                                                PointerWrapperT in_crows_vec,
                                                IndexT* rows_nnzs) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_rows, IndexT) {
    if (tid_x == 0) {
      rows_nnzs[0] = 0;
    }
    IndexT now_rows = tid_x / in_num;
    // index mean the number of input  tensor
    IndexT index = tid_x % in_num;

    rows_nnzs[tid_x + 1] =
        in_crows_vec[index][now_rows + 1] - in_crows_vec[index][now_rows];
  }
}

template <typename T,
          typename IndexT,
          typename PointerWrapperT,
          typename PointerWrapperIndexT,
          typename DarrayWrapperT>
__global__ void ConcatCsr2D1ASetValueKernel(const IndexT total_nnz,
                                            const size_t in_num,
                                            const IndexT rows,
                                            PointerWrapperT in_values,
                                            PointerWrapperIndexT in_cols,
                                            PointerWrapperIndexT in_crows,
                                            IndexT* rows_nnzs_offsets,
                                            DarrayWrapperT col_offsets,
                                            T* out_values,
                                            IndexT* out_cols) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_nnz, IndexT) {
    IndexT i = phi::funcs::UpperBound<IndexT, IndexT>(
        rows_nnzs_offsets, rows * in_num + 1, tid_x);
    i--;
    IndexT left_nnz = tid_x - rows_nnzs_offsets[i];

    IndexT index = i % in_num;
    IndexT now_rows = i / in_num;

    IndexT total_offset = left_nnz + in_crows[index][now_rows];

    out_values[tid_x] = in_values[index][total_offset];
    // need to add the previous tensor's col number in this line
    out_cols[tid_x] = in_cols[index][total_offset] + col_offsets[index];
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr2D1ASetCrowsKernel(const IndexT rows,
                                            const size_t in_num,
                                            PointerWrapperT in_crows_vec,
                                            IndexT* out_crows) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, in_num, IndexT) {
    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < rows; tid_y += blockDim.y * gridDim.y) {
      phi::CudaAtomicAdd(&out_crows[tid_y + 1], in_crows_vec[tid_x][tid_y + 1]);
    }
  }
}

template <typename IndexT, typename PointerWrapperT, typename DarrayWrapperT>
__global__ void ConcatCsr3D1AGetHelpArrayKernel(

    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_crows_vec,
    DarrayWrapperT rows_numel,
    IndexT* out_crows_offset,

    IndexT* values_index_offset,
    IndexT* batch_nnz) {
  // tid_x = batch  tid_y = the number of input  tensor

  CUDA_KERNEL_LOOP_TYPE(tid_x, batch, IndexT) {
    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < in_num; tid_y += blockDim.y * gridDim.y) {
      IndexT pos = (tid_x + 1) * (rows_numel[tid_y] + 1) - 1;
      // out_crows_offset and values_index_offset are 2D array like
      // out_crows_offset[batch][in_num + 1]
      out_crows_offset[tid_x * (in_num + 1) + tid_y + 1] =
          in_crows_vec[tid_y][pos];

      values_index_offset[tid_x * (in_num) + tid_y + 1] =
          in_crows_vec[tid_y][pos];

      batch_nnz[tid_y * (batch + 1) + tid_x + 1] = in_crows_vec[tid_y][pos];

      __syncthreads();
      if (tid_y == 0) {
        // accumulate the first dimension (across different batches) of the 2D
        // array out_crows_offset.
        for (int i = 0; i < in_num - 1; i++) {
          out_crows_offset[tid_x * (in_num + 1) + i + 1] +=
              out_crows_offset[tid_x * (in_num + 1) + i];
        }
      }
    }
  }
}

template <typename IndexT, typename PointerWrapperT, typename DarrayWrapperT>
__global__ void ConcatCsr3D1ASetCrowsKernel(const IndexT in_num,
                                            const IndexT batch,
                                            const IndexT total_rows,
                                            PointerWrapperT in_crows_vec,
                                            DarrayWrapperT rows_nums,
                                            DarrayWrapperT in_rows_offsets,
                                            IndexT* out_crows_offset,
                                            IndexT* out_crows) {
  // tid_y== batch  tid_z == index tid_x == rows
  IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  IndexT tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  IndexT tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  for (; tid_z < in_num; tid_z += blockDim.z * gridDim.z) {
    for (; tid_y < batch; tid_y += blockDim.y * gridDim.y) {
      CUDA_KERNEL_LOOP_TYPE(tid_x, rows_nums[tid_z], IndexT) {
        IndexT in_crows_pos = tid_y * (rows_nums[tid_z] + 1) + tid_x + 1;

        IndexT out_pos =
            tid_y * (total_rows + 1) + in_rows_offsets[tid_z] + tid_x + 1;

        IndexT offset = out_crows_offset[tid_y * (in_num + 1) + tid_z];

        out_crows[out_pos] = in_crows_vec[tid_z][in_crows_pos] + offset;

        if (tid_x == 0 && tid_z == 0) {
          out_crows[tid_y * (total_rows + 1)] = 0;
        }
      }
    }
  }
}

template <typename T,
          typename IndexT,
          typename DarrayWrapperT,
          typename PointerWrapperT,
          typename PointerWrapperIndexT>
__global__ void ConcatCsr3D1ASetValuesColsKernel(
    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_values_data,
    PointerWrapperIndexT in_cols_data,
    DarrayWrapperT in_nnz,
    IndexT* values_index_offset,
    IndexT* batch_nnz,
    T* out_values,
    IndexT* out_cols) {
  // tid_x == in tensor tid_y == all nnz for in tensor
  IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  CUDA_KERNEL_LOOP_TYPE(tid_x, in_num, IndexT) {
    for (; tid_y < in_nnz[tid_x]; tid_y += blockDim.y * gridDim.y) {
      // Each computation of b is only computed within the corresponding tensor
      // batch_nnz itself is based on a two-dimensional structure like
      // batch_nnz[i][b].

      IndexT b = 0;
      IndexT* index_nnz_ptr = batch_nnz + tid_x * (batch + 1);
      IndexT next_offset = index_nnz_ptr[b + 1];
      IndexT curr_offset = index_nnz_ptr[b];

      while (next_offset <= tid_y) {
        curr_offset = next_offset;
        ++b;
        next_offset += index_nnz_ptr[b + 1];
      }

      IndexT left_nnz = tid_y - curr_offset;
      IndexT pos = values_index_offset[b * (in_num) + tid_x];

      out_values[pos + left_nnz] = in_values_data[tid_x][tid_y];
      out_cols[pos + left_nnz] = in_cols_data[tid_x][tid_y];
    }
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D2ASetCrowsKernel(const IndexT crows_num,
                                            const size_t in_num,
                                            const IndexT batch,
                                            const IndexT rows,
                                            PointerWrapperT in_crows_vec,
                                            IndexT* out_crows) {
  IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  for (; tid_y < in_num; tid_y += blockDim.y * gridDim.y) {
    IndexT tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid_x < crows_num * batch; tid_x += blockDim.x * gridDim.x) {
      phi::CudaAtomicAdd(&out_crows[tid_x], in_crows_vec[tid_y][tid_x]);
    }
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D2AGetHelpArrayKernel(const IndexT rows,
                                                const size_t in_num,
                                                const IndexT batch,
                                                PointerWrapperT in_crows_vec,
                                                IndexT* rows_nnz,
                                                IndexT* in_index_batch_nnz) {
  IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  IndexT tid_z = blockIdx.z * blockDim.z + threadIdx.z;

  for (; tid_z < in_num; tid_z += blockDim.z * gridDim.z) {
    for (; tid_y < batch; tid_y += blockDim.y * gridDim.y) {
      CUDA_KERNEL_LOOP_TYPE(tid_x, rows, IndexT) {
        IndexT curr_offset = tid_y * (rows + 1) + tid_x;

        IndexT now = rows * in_num * tid_y + in_num * tid_x + tid_z;
        rows_nnz[now] = in_crows_vec[tid_z][curr_offset + 1] -
                        in_crows_vec[tid_z][curr_offset];
      }
    }
  }
  __syncthreads();
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D2AGetBatchNnzArrayKernel(
    const IndexT rows,
    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_crows_vec,
    IndexT* in_batch_offsets,
    IndexT* in_index_batch_nnz) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, batch, IndexT) {
    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    IndexT now = (tid_x + 1) * (rows + 1) - 1;

    for (; tid_y < in_num; tid_y += blockDim.y * gridDim.y) {
      phi::CudaAtomicAdd(&in_batch_offsets[tid_x + 1],
                         in_crows_vec[tid_y][now]);

      // in_index_batch_nnz is 2D array in_index_batch_nnz[in num][batch + 1]
      in_index_batch_nnz[tid_y * (batch + 1) + tid_x + 1] =
          in_crows_vec[tid_y][now];

      __syncthreads();

      if (tid_x == 0) {
        // accumulate the first dimension (across different in num) of the 2D
        // array in_index_batch_nnz.
        for (int i = 0; i < batch; i++) {
          in_index_batch_nnz[tid_y * (batch + 1) + i + 1] +=
              in_index_batch_nnz[tid_y * (batch + 1) + i];
        }
      }
    }
  }
}

template <typename T,
          typename IndexT,
          typename DarrayWrapperT,
          typename PointerWrapperT,
          typename PointerWrapperIndexT>
__global__ void ConcatCsr3D2ASetvaluesKernel(const size_t in_num,
                                             const int rows,
                                             const int batch,
                                             const int64_t total_nnz,
                                             PointerWrapperT in_values_vec,
                                             PointerWrapperIndexT in_cols_vec,
                                             PointerWrapperIndexT in_crows_vec,
                                             IndexT* rows_nnz,
                                             IndexT* in_batch_offsets,
                                             IndexT* in_index_batch_nnz,
                                             DarrayWrapperT d_col_offsets,
                                             T* out_values,
                                             IndexT* out_cols) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_nnz, IndexT) {
    IndexT b = phi::funcs::UpperBound<IndexT, IndexT>(
        in_batch_offsets, batch + 1, tid_x);
    b--;
    IndexT left_nnz = tid_x - in_batch_offsets[b];
    IndexT* now_batch_nnz = rows_nnz + rows * in_num * b;
    IndexT i = 0;
    IndexT curr_offset = 0;
    IndexT next_offset = now_batch_nnz[i];

    while (next_offset <= left_nnz) {
      curr_offset = next_offset;
      ++i;
      next_offset += now_batch_nnz[i];
    }
    IndexT left_nnz2 = left_nnz - curr_offset;
    IndexT index = i % in_num;
    IndexT now_rows = i / in_num;
    IndexT total_offset = left_nnz2 +
                          in_crows_vec[index][now_rows + (rows + 1) * b] +
                          in_index_batch_nnz[index * (batch + 1) + b];
    out_values[tid_x] = in_values_vec[index][total_offset];

    out_cols[tid_x] = in_cols_vec[index][total_offset] + d_col_offsets[index];
  }
}

template <typename T, typename Context>
void ConcatCooGPUKernel(const Context& dev_ctx,
                        const std::vector<const SparseCooTensor*>& x,
                        const Scalar& axis_scalar,
                        SparseCooTensor* out) {
  std::vector<DenseTensor> indices;
  std::vector<DenseTensor> values;
  std::vector<phi::DDim> x_dims;
  int64_t in_num = x.size();
  int64_t axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());
  int64_t sparse_dim = x[0]->sparse_dim();
  int64_t dense_dim = x[0]->dense_dim();

  DDim dims = x[0]->dims();

  int64_t pos = 0;
  for (const auto* t : x) {
    check_cat_sparse_dims(t, pos, dims, axis, sparse_dim, dense_dim);
    x_dims.push_back(t->dims());
    pos++;
  }

  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  if (axis < sparse_dim) {
    int64_t out_nnz = 0, out_cols = 0;
    std::vector<int64_t> indice_offsets;
    std::vector<int64_t> nnz_offsets;
    nnz_offsets.push_back(0);
    indice_offsets.push_back(0);
    for (const auto* t : x) {
      indices.emplace_back(t->indices());
      values.emplace_back(t->values());
      out_nnz += t->nnz();
      nnz_offsets.push_back(out_nnz);
      out_cols += t->dims()[axis];
      indice_offsets.push_back(out_cols);
    }

    DenseTensor out_indices;
    DenseTensor out_values;
    DDim v_dim = x[0]->values().dims();
    v_dim[0] = out_nnz;
    IntArray v_shape(v_dim.GetMutable(), v_dim.size());
    out_values = phi::Empty<T, Context>(dev_ctx, v_shape);
    out_indices = phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_nnz});
    int64_t* out_indices_data = out_indices.data<int64_t>();

    phi::Allocator::AllocationPtr dev_indice_offsets_ptr{nullptr};
    DArray<int64_t> d_indice_offsets(
        dev_ctx, indice_offsets, &dev_indice_offsets_ptr);
    int64_t* d_indice_offsets_ptr = d_indice_offsets.get_ptr();
    phi::Allocator::AllocationPtr dev_nnz_offsets_ptr{nullptr};
    DArray<int64_t> d_nnz_offsets(dev_ctx, nnz_offsets, &dev_nnz_offsets_ptr);
    int64_t* d_nnz_offsets_ptr = d_nnz_offsets.get_ptr();

    phi::sparse::ConcatFunctor<int64_t, Context>(
        dev_ctx, indices, static_cast<int>(1), &out_indices);
    phi::sparse::ConcatFunctor<T, Context>(
        dev_ctx, values, static_cast<int>(0), &out_values);

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz, 1);

    ConcatCooSetIndicesKernel<int64_t>
        <<<config.block_per_grid.x,
           config.thread_per_block.x,
           0,
           dev_ctx.stream()>>>(out_nnz,
                               axis,
                               in_num,
                               d_indice_offsets_ptr,
                               d_nnz_offsets_ptr,
                               out_indices_data);
    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());
  } else {
    int64_t values_dim = axis - sparse_dim + 1;
    int64_t total_size = 0;
    for (auto& r : x) {
      total_size += r->values().dims()[values_dim];
    }
    DDim zeros_sizes = x[0]->values().dims();
    int64_t cumulative_size = 0;

    for (const auto* t : x) {
      zeros_sizes[0] = t->values().dims()[0];
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t->values().dims()[values_dim];
      // z1 z2是全0的向量
      DenseTensor z1 =
          phi::Full<T, Context>(dev_ctx, common::vectorize(zeros_sizes), 0);
      zeros_sizes[values_dim] = total_size - cumulative_size;
      DenseTensor z2 =
          phi::Full<T, Context>(dev_ctx, common::vectorize(zeros_sizes), 0);
      std::vector<DenseTensor> now_values;
      now_values.push_back(z1);
      now_values.push_back(t->values());
      now_values.push_back(z2);
      zeros_sizes[values_dim] = total_size;

      DenseTensor concat_value =
          phi::Empty<T, Context>(dev_ctx, common::vectorize(zeros_sizes));
      phi::sparse::ConcatFunctor<T, Context>(
          dev_ctx, now_values, values_dim, &concat_value);

      values.push_back(concat_value);
      indices.push_back(t->indices());
    }
    zeros_sizes[values_dim] = total_size * x.size();
    DenseTensor out_values =
        phi::Empty<T, Context>(dev_ctx, common::vectorize(zeros_sizes));
    DenseTensor out_indices =
        phi::Empty<int64_t, Context>(dev_ctx, common::vectorize(zeros_sizes));
    phi::sparse::ConcatFunctor<int64_t, Context>(
        dev_ctx, indices, static_cast<int>(1), &out_indices);
    phi::sparse::ConcatFunctor<T, Context>(
        dev_ctx, values, static_cast<int>(0), &out_values);

    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());
  }
}

template <typename T, typename Context>
void ConcatCooKernel(const Context& dev_ctx,
                     const std::vector<const SparseCooTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCooTensor* out) {
  ConcatCooGPUKernel<T, Context>(dev_ctx, x, axis_scalar, out);
}

template <typename T, typename IndexT, typename Context>
void ConcatCsrGPU2D0A(const Context& dev_ctx,
                      const std::vector<const SparseCsrTensor*>& x,
                      const size_t in_num,
                      const int64_t out_values_size,
                      const phi::DDim& out_dims,
                      const std::vector<const T*>& values_vec,
                      const std::vector<const int64_t*>& cols_vec,
                      const std::vector<const int64_t*>& crows_vec,
                      SparseCsrTensor* out) {
  std::vector<int64_t> in_rows;
  std::vector<int64_t> in_rows_offsets;
  std::vector<int64_t> nnz_offset_array;
  in_rows.push_back(0);
  in_rows_offsets.push_back(0);
  int64_t rows_offset = 0;
  int64_t rows = 0;
  int64_t nnz_offset = 0;
  nnz_offset_array.push_back(nnz_offset);
  for (auto t : x) {
    rows = static_cast<int>(t->dims()[0]);
    rows_offset += rows;
    in_rows.push_back(rows);
    in_rows_offsets.push_back(rows_offset);

    nnz_offset += t->nnz();
    nnz_offset_array.push_back(nnz_offset);
  }
  // The length of out_crows is equal to the sum of the rows of all input
  // tensors plus 1, where the 1 represents the initial 0. Therefore, the last
  // rows_offset will be used as the length of out_crows
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {rows_offset + 1});
  int64_t* d_out_crows = out_crows.data<int64_t>();

  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});

  auto gpu_place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();

  T* d_out_values = out_values.data<T>();
  int64_t* d_out_cols = out_cols.data<int64_t>();

  const int64_t* const* in_crows_vec = crows_vec.data();
  phi::Allocator::AllocationPtr dev_ins_ptr{nullptr};
  PointerToPointer<int64_t> d_in_crows_vec(
      dev_ctx, in_num, in_crows_vec, &dev_ins_ptr);
  phi::Allocator::AllocationPtr dev_vec_vec_ptr{nullptr};
  const T* const* values_vec_data = values_vec.data();
  PointerToPointer<T> d_in_values_vec(
      dev_ctx, in_num, values_vec_data, &dev_vec_vec_ptr);
  phi::Allocator::AllocationPtr dev_cols_vec_ptr{nullptr};
  const int64_t* const* cols_vec_data = cols_vec.data();
  PointerToPointer<int64_t> d_in_cols_vec(
      dev_ctx, in_num, cols_vec_data, &dev_cols_vec_ptr);

  // d_in_rows means the rows for each tensor
  phi::Allocator::AllocationPtr dev_crows_pos_ptr{nullptr};
  DArray<int64_t> d_in_rows(dev_ctx, in_rows, &dev_crows_pos_ptr);

  phi::Allocator::AllocationPtr dev_rows_ptr{nullptr};
  DArray<int64_t> d_in_rows_offsets(dev_ctx, in_rows_offsets, &dev_rows_ptr);
  // number of nnz  for each row, it will be accumulated to facilitate the
  // calculation of d_out_crows
  DenseTensor out_crows_offsets =
      phi::Empty<int64_t>(dev_ctx, {static_cast<int64_t>(in_num + 1)});
  int64_t* d_out_crows_offsets = out_crows_offsets.data<int64_t>();

  phi::Allocator::AllocationPtr dev_nnz_offsets_pos_ptr{nullptr};
  DArray<int64_t> d_nnz_offsets(
      dev_ctx, nnz_offset_array, &dev_nnz_offsets_pos_ptr);

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, in_num, 1);
  ConcatCsr2D0AGetHelpArrayKernel<int64_t,
                                  decltype(d_in_crows_vec),
                                  decltype(d_in_rows)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(
          in_num, d_in_crows_vec, d_in_rows, d_out_crows_offsets);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, rows_offset, 1);

  ConcatCsr2D0ASetCrowsKernel<int64_t,
                              decltype(d_in_crows_vec),
                              decltype(d_in_rows_offsets)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(rows_offset,
                             in_num,
                             d_in_crows_vec,
                             d_in_rows_offsets,
                             d_out_crows_offsets,
                             d_out_crows);

  // nnz_offset = sum of nnz
  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, nnz_offset, 1);

  ConcatCsr2D0ASetValueKernel<T,
                              int64_t,
                              decltype(d_in_values_vec),
                              decltype(d_in_cols_vec),
                              decltype(d_nnz_offsets)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(nnz_offset,
                             in_num,
                             d_in_values_vec,
                             d_in_cols_vec,
                             d_nnz_offsets,
                             d_out_values,
                             d_out_cols);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename IndexT, typename Context>
void ConcatCsrGPU2D1A(const Context& dev_ctx,
                      const std::vector<const SparseCsrTensor*>& x,
                      size_t in_num,
                      const int64_t out_values_size,
                      const phi::DDim& out_dims,

                      const std::vector<const T*>& values_vec,
                      const std::vector<const int64_t*>& cols_vec,
                      const std::vector<const int64_t*>& crows_vec,
                      SparseCsrTensor* out) {
  std::vector<int64_t> col_offsets;

  int64_t col_offset = 0;
  int64_t total_nnz = 0;
  col_offsets.push_back(col_offset);
  for (const auto* t : x) {
    col_offset += static_cast<int64_t>(t->dims()[1]);
    col_offsets.push_back(col_offset);
    total_nnz += t->nnz();
  }
  int64_t rows = static_cast<size_t>(x[0]->dims()[0]);
  int64_t out_crows_size = rows + 1;
  int64_t total_rows = rows * in_num;

  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});
  int64_t* d_out_crows = out_crows.data<int64_t>();

  phi::funcs::SetConstant<GPUContext, int64_t> set_zero;
  set_zero(dev_ctx, &out_crows, 0);

  T* d_out_values = out_values.data<T>();
  int64_t* d_out_cols = out_cols.data<int64_t>();

  // the number of nnz in each row
  DenseTensor rows_nnzs_tensor = phi::Empty<int64_t>(dev_ctx, {total_rows + 1});
  int64_t* d_rows_nnzs = rows_nnzs_tensor.data<int64_t>();

  phi::Allocator::AllocationPtr dev_crows_vec_ptr{nullptr};
  const int64_t* const* crows_vec_data = crows_vec.data();
  PointerToPointer<int64_t> d_in_crows_vec(
      dev_ctx, in_num, crows_vec_data, &dev_crows_vec_ptr);
  phi::Allocator::AllocationPtr dev_vec_vec_ptr{nullptr};
  const T* const* values_vec_data = values_vec.data();
  PointerToPointer<T> d_in_values_vec(
      dev_ctx, in_num, values_vec_data, &dev_vec_vec_ptr);
  phi::Allocator::AllocationPtr dev_cols_vec_ptr{nullptr};
  const int64_t* const* cols_vec_data = cols_vec.data();
  PointerToPointer<int64_t> d_in_cols_vec(
      dev_ctx, in_num, cols_vec_data, &dev_cols_vec_ptr);

  // Number of col in each tensor
  phi::Allocator::AllocationPtr dev_col_offsets_ptr{nullptr};
  DArray<int64_t> d_col_offsets(dev_ctx, col_offsets, &dev_col_offsets_ptr);

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total_rows, 1);
  ConcatCsr2D1AGetHelpArrayKernel<int64_t, decltype(d_in_crows_vec)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(
          total_rows, in_num, rows, d_in_crows_vec, d_rows_nnzs);

  thrust::inclusive_scan(
      thrust::device_pointer_cast(d_rows_nnzs),
      thrust::device_pointer_cast(d_rows_nnzs) + total_rows + 1,
      d_rows_nnzs);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total_nnz, 1);
  ConcatCsr2D1ASetValueKernel<T,
                              int64_t,
                              decltype(d_in_values_vec),
                              decltype(d_in_cols_vec),
                              decltype(d_col_offsets)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(total_nnz,
                             in_num,
                             rows,
                             d_in_values_vec,
                             d_in_cols_vec,
                             d_in_crows_vec,
                             d_rows_nnzs,
                             d_col_offsets,
                             d_out_values,
                             d_out_cols);

  config = phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, in_num, rows);

  ConcatCsr2D1ASetCrowsKernel<int64_t, decltype(d_in_crows_vec)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          rows, in_num, d_in_crows_vec, d_out_crows);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename IndexT, typename Context>
void ConcatCsrGPU3D0A(const Context& dev_ctx,
                      const std::vector<const SparseCsrTensor*>& x,
                      size_t in_num,
                      const int64_t out_values_size,
                      const phi::DDim& out_dims,
                      const std::vector<const T*>& values_vec,
                      const std::vector<const int64_t*>& cols_vec,
                      const std::vector<const int64_t*>& crows_vec,
                      SparseCsrTensor* out) {
  std::vector<DenseTensor> crows;
  std::vector<DenseTensor> values;
  std::vector<DenseTensor> cols;
  int64_t out_crows_size = 0;
  for (size_t i = 0; i < in_num; i++) {
    crows.emplace_back(x[i]->crows());
    values.emplace_back(x[i]->values());
    cols.emplace_back(x[i]->cols());
    out_crows_size += x[i]->crows().numel();
  }
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});
  phi::sparse::ConcatFunctor<T, Context>(
      dev_ctx, values, static_cast<T>(0), &out_values);
  phi::sparse::ConcatFunctor<int64_t, Context>(
      dev_ctx, cols, static_cast<int64_t>(0), &out_cols);
  phi::sparse::ConcatFunctor<int64_t, Context>(
      dev_ctx, crows, static_cast<int64_t>(0), &out_crows);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename IndexT, typename Context>
void ConcatCsrGPU3D1A(const Context& dev_ctx,
                      const std::vector<const SparseCsrTensor*>& x,
                      size_t in_num,
                      const int64_t out_values_size,
                      const phi::DDim& out_dims,

                      const std::vector<const T*>& values_vec,
                      const std::vector<const int64_t*>& cols_vec,
                      const std::vector<const int64_t*>& crows_vec,
                      SparseCsrTensor* out) {
  size_t batch = static_cast<int>(x[0]->dims()[0]);

  int64_t out_crows_size = batch;
  std::vector<int64_t> rows_nums;
  std::vector<int64_t> rows_offsets;  // 每一个index下的rows的叠加值
  std::vector<int64_t> in_nnz;
  int64_t rows_offset = 0;
  int64_t max_rows = 0;
  int64_t max_nnz = 0;

  rows_offsets.push_back(rows_offset);
  for (size_t i = 0; i < in_num; i++) {
    int64_t rows = static_cast<int64_t>(x[i]->dims()[1]);
    rows_nums.push_back(rows);
    out_crows_size += batch * rows;
    rows_offset += rows;
    rows_offsets.push_back(rows_offset);
    max_rows = max_rows > rows ? max_rows : rows;
    int64_t nnz = static_cast<int64_t>(x[i]->nnz());
    in_nnz.push_back(nnz);
    max_nnz = max_nnz > nnz ? max_nnz : nnz;
  }
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});
  int64_t* d_out_crows = out_crows.data<int64_t>();
  T* d_out_values = out_values.data<T>();
  int64_t* d_out_cols = out_cols.data<int64_t>();

  phi::Allocator::AllocationPtr dev_crows_vec_ptr{nullptr};
  const int64_t* const* crows_vec_data = crows_vec.data();
  PointerToPointer<int64_t> d_in_crows_vec(
      dev_ctx, in_num, crows_vec_data, &dev_crows_vec_ptr);
  phi::Allocator::AllocationPtr dev_vec_vec_ptr{nullptr};
  const T* const* values_vec_data = values_vec.data();
  PointerToPointer<T> d_in_values_vec(
      dev_ctx, in_num, values_vec_data, &dev_vec_vec_ptr);
  phi::Allocator::AllocationPtr dev_cols_vec_ptr{nullptr};
  const int64_t* const* cols_vec_data = cols_vec.data();
  PointerToPointer<int64_t> d_in_cols_vec(
      dev_ctx, in_num, cols_vec_data, &dev_cols_vec_ptr);

  phi::funcs::SetConstant<GPUContext, int64_t> set_zero;

  phi::Allocator::AllocationPtr dev_rows_nums_ptr{nullptr};
  DArray<int64_t> d_rows_nums(dev_ctx, rows_nums, &dev_rows_nums_ptr);
  phi::Allocator::AllocationPtr dev_rows_offsets_ptr{nullptr};
  DArray<int64_t> d_rows_offsets(dev_ctx, rows_offsets, &dev_rows_offsets_ptr);
  phi::Allocator::AllocationPtr dev_in_nnz_ptr{nullptr};
  DArray<int64_t> d_in_nnz(dev_ctx, in_nnz, &dev_in_nnz_ptr);

  DenseTensor out_crows_offset_tensor = phi::Empty<int64_t>(
      dev_ctx, {static_cast<int64_t>(batch * (in_num + 1))});
  int64_t* d_out_crows_offset = out_crows_offset_tensor.data<int64_t>();
  set_zero(dev_ctx, &out_crows_offset_tensor, 0);
  DenseTensor values_index_offset_tensor =
      phi::Empty<int64_t>(dev_ctx, {static_cast<int64_t>(batch * in_num + 1)});
  int64_t* d_values_index_offset = values_index_offset_tensor.data<int64_t>();
  set_zero(dev_ctx, &values_index_offset_tensor, 0);
  DenseTensor batch_nnz_tensor = phi::Empty<int64_t>(
      dev_ctx, {static_cast<int64_t>(in_num * (batch + 1))});
  // batch_nnz is the array of nnz corresponding to batch and number of tensor.
  // but because it's necessary to have number of tensor as one dimension and
  // batch as two. batch_nnz[i][b]
  int64_t* d_batch_nnz = batch_nnz_tensor.data<int64_t>();
  set_zero(dev_ctx, &batch_nnz_tensor, 0);
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, batch, in_num);
  ConcatCsr3D1AGetHelpArrayKernel<int64_t,
                                  decltype(d_in_crows_vec),
                                  decltype(d_rows_nums)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          in_num,
          batch,
          d_in_crows_vec,
          d_rows_nums,
          d_out_crows_offset,
          d_values_index_offset,
          d_batch_nnz);

  thrust::inclusive_scan(
      thrust::device_pointer_cast(d_values_index_offset),
      thrust::device_pointer_cast(d_values_index_offset) + batch * in_num + 1,
      d_values_index_offset);

  config = phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, in_num, max_nnz);

  ConcatCsr3D1ASetValuesColsKernel<T,
                                   int64_t,
                                   decltype(d_in_nnz),
                                   decltype(d_in_values_vec),
                                   decltype(d_in_cols_vec)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          in_num,
          batch,
          d_in_values_vec,
          d_in_cols_vec,
          d_in_nnz,
          d_values_index_offset,
          d_batch_nnz,
          d_out_values,
          d_out_cols);

  // note: The order of GetGpuLaunchConfig3D is z, y , x

  config = phi::backends::gpu::GetGpuLaunchConfig3D2(
      dev_ctx, max_rows, batch, in_num);

  ConcatCsr3D1ASetCrowsKernel<int64_t,
                              decltype(d_in_crows_vec),
                              decltype(d_rows_offsets)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          in_num,
          batch,
          rows_offset,
          d_in_crows_vec,
          d_rows_nums,
          d_rows_offsets,
          d_out_crows_offset,
          d_out_crows);
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename IndexT, typename Context>
void ConcatCsrGPU3D2A(const Context& dev_ctx,
                      const std::vector<const SparseCsrTensor*>& x,
                      size_t in_num,
                      const int64_t out_values_size,
                      const phi::DDim& out_dims,

                      const std::vector<const T*>& values_vec,
                      const std::vector<const int64_t*>& cols_vec,
                      const std::vector<const int64_t*>& crows_vec,
                      SparseCsrTensor* out) {
  auto batch = static_cast<int>(x[0]->dims()[0]);
  auto rows = static_cast<int>(x[0]->dims()[1]);
  auto now_crow_numel = rows + 1;

  int64_t total_nnz = 0;
  int64_t col_offset = 0;
  std::vector<int64_t> col_offsets;
  col_offsets.push_back(col_offset);
  for (auto* t : x) {
    total_nnz += static_cast<int64_t>(t->nnz());
    col_offset += static_cast<int64_t>(t->dims()[2]);
    col_offsets.push_back(col_offset);
  }

  DenseTensor out_crows =
      phi::Empty<int64_t>(dev_ctx, {now_crow_numel * batch});
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});

  int64_t* d_out_crows = out_crows.data<int64_t>();

  phi::funcs::SetConstant<GPUContext, int64_t> set_zero;
  set_zero(dev_ctx, &out_crows, 0);

  T* d_out_values = out_values.data<T>();
  int64_t* d_out_cols = out_cols.data<int64_t>();

  phi::Allocator::AllocationPtr dev_crows_vec_ptr{nullptr};
  const int64_t* const* crows_vec_data = crows_vec.data();
  PointerToPointer<int64_t> d_in_crows_vec(
      dev_ctx, in_num, crows_vec_data, &dev_crows_vec_ptr);
  phi::Allocator::AllocationPtr dev_vec_vec_ptr{nullptr};
  const T* const* values_vec_data = values_vec.data();
  PointerToPointer<T> d_in_values_vec(
      dev_ctx, in_num, values_vec_data, &dev_vec_vec_ptr);
  phi::Allocator::AllocationPtr dev_cols_vec_ptr{nullptr};
  const int64_t* const* cols_vec_data = cols_vec.data();
  PointerToPointer<int64_t> d_in_cols_vec(
      dev_ctx, in_num, cols_vec_data, &dev_cols_vec_ptr);

  phi::Allocator::AllocationPtr dev_col_offsets_ptr{nullptr};
  DArray<int64_t> d_col_offsets(dev_ctx, col_offsets, &dev_col_offsets_ptr);

  DenseTensor rows_nnz_tensor = phi::Empty<int64_t>(
      dev_ctx, {static_cast<int64_t>(in_num * batch * rows)});
  // // number of nnz for per rows
  int64_t* d_rows_nnz = rows_nnz_tensor.data<int64_t>();

  DenseTensor in_index_batch_tensor = phi::Empty<int64_t>(
      dev_ctx, {static_cast<int64_t>(in_num * (batch + 1))});
  int64_t* d_in_index_batch_nnz = in_index_batch_tensor.data<int64_t>();
  set_zero(dev_ctx, &in_index_batch_tensor, 0);
  // 每一轮batch中含有的nnz数量的和的
  DenseTensor in_batch_offsets_tensor =
      phi::Empty<int64_t>(dev_ctx, {batch + 1});
  int64_t* d_in_batch_offsets = in_batch_offsets_tensor.data<int64_t>();
  set_zero(dev_ctx, &in_batch_offsets_tensor, 0);

  auto config = phi::backends::gpu::GetGpuLaunchConfig2D(
      dev_ctx, now_crow_numel * batch, in_num);
  ConcatCsr3D2ASetCrowsKernel<int64_t, decltype(d_in_crows_vec)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          now_crow_numel, in_num, batch, rows, d_in_crows_vec, d_out_crows);

  config =
      phi::backends::gpu::GetGpuLaunchConfig3D2(dev_ctx, rows, batch, in_num);
  ConcatCsr3D2AGetHelpArrayKernel<int64_t, decltype(d_in_crows_vec)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          rows,
          in_num,
          batch,
          d_in_crows_vec,
          d_rows_nnz,
          d_in_index_batch_nnz);

  config = phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, batch, in_num);
  ConcatCsr3D2AGetBatchNnzArrayKernel<int64_t, decltype(d_in_crows_vec)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          rows,
          in_num,
          batch,
          d_in_crows_vec,
          d_in_batch_offsets,
          d_in_index_batch_nnz);

  thrust::inclusive_scan(
      thrust::device_pointer_cast(d_in_batch_offsets),
      thrust::device_pointer_cast(d_in_batch_offsets) + batch + 1,
      d_in_batch_offsets);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total_nnz, 1);
  ConcatCsr3D2ASetvaluesKernel<T,
                               int64_t,
                               decltype(d_col_offsets),
                               decltype(d_in_values_vec),
                               decltype(d_in_cols_vec)>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          in_num,
          rows,
          batch,
          total_nnz,
          d_in_values_vec,
          d_in_cols_vec,
          d_in_crows_vec,
          d_rows_nnz,
          d_in_batch_offsets,
          d_in_index_batch_nnz,
          d_col_offsets,
          d_out_values,
          d_out_cols);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void ConcatCsrGPUKernel(const Context& dev_ctx,
                        const std::vector<const SparseCsrTensor*>& x,
                        const Scalar& axis_scalar,
                        SparseCsrTensor* out) {
  size_t in_num = x.size();

  int64_t axis = axis_scalar.to<int64_t>();

  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());

  std::vector<phi::DDim> x_dims;
  x_dims.reserve(in_num);
  std::vector<int64_t> crows_numel;
  std::vector<const int64_t*> crows_vec;
  std::vector<const T*> values_vec;
  std::vector<const int64_t*> cols_vec;

  int64_t out_values_size = 0;
  int64_t out_crows_size = 0;
  for (const auto* t : x) {
    x_dims.emplace_back(t->dims());
    values_vec.push_back(t->values().data<T>());
    cols_vec.push_back(t->cols().data<int64_t>());
    crows_vec.push_back(t->crows().data<int64_t>());
    out_values_size += t->nnz();
  }

  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  int x_dim = x_dims[0].size();
  if (x_dim == 2) {
    if (axis == 0) {
      ConcatCsrGPU2D0A<T, int64_t, Context>(dev_ctx,
                                            x,
                                            in_num,
                                            out_values_size,
                                            out_dims,
                                            values_vec,
                                            cols_vec,
                                            crows_vec,
                                            out);
    } else {
      ConcatCsrGPU2D1A<T, int64_t, Context>(dev_ctx,
                                            x,
                                            in_num,
                                            out_values_size,
                                            out_dims,

                                            values_vec,
                                            cols_vec,
                                            crows_vec,
                                            out);
    }

  } else if (x_dim == 3) {
    if (axis == 0) {
      ConcatCsrGPU3D0A<T, int64_t, Context>(dev_ctx,
                                            x,
                                            in_num,
                                            out_values_size,
                                            out_dims,

                                            values_vec,
                                            cols_vec,
                                            crows_vec,
                                            out);
    } else if (axis == 1) {
      ConcatCsrGPU3D1A<T, int64_t, Context>(dev_ctx,
                                            x,
                                            in_num,
                                            out_values_size,
                                            out_dims,

                                            values_vec,
                                            cols_vec,
                                            crows_vec,
                                            out);
    } else {
      ConcatCsrGPU3D2A<T, int64_t, Context>(dev_ctx,
                                            x,
                                            in_num,
                                            out_values_size,
                                            out_dims,

                                            values_vec,
                                            cols_vec,
                                            crows_vec,
                                            out);
    }
  } else {
    // throw exception
    phi::errors::InvalidArgument(
        "Concat for Sparse CSR Tensor only support 2-D or 3-D, but got %d-D.",
        x_dims.size());
  }
}

template <typename T, typename Context>
void ConcatCsrKernel(const Context& dev_ctx,
                     const std::vector<const SparseCsrTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCsrTensor* out) {
  ConcatCsrGPUKernel<T, Context>(dev_ctx, x, axis_scalar, out);
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(concat_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCooKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}

PD_REGISTER_KERNEL(concat_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCsrKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}
