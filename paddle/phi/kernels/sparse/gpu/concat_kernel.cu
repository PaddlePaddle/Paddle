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
#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename IndexT>
struct DArray {
 public:
  IndexT* d_array{nullptr};

  __device__ inline const void* operator[](int i) const { return d_array[i]; }

  DArray() = default;

  DArray(const phi::GPUContext& ctx, const std::vector<IndexT>& host_array) {
    // copy offsets to device
    auto d_array_tensor = memory_utils::Alloc(
        dev_ctx.GetPlace(),
        sizeof(IndexT) * host_array.size(),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    d_array = reinterpret_cast<IndexT*>(d_array_tensor->ptr());

    memory_utils::Copy(dev_ctx.GetPlace(),
                       d_array,
                       phi::CPUPlace(),
                       host_array.data(),
                       sizeof(IndexT) * host_array.size(),
                       dev_ctx.stream());
  }
}

template <typename T>
struct PointerToPointer {
 public:
  void** ins_addr{nullptr};
  __device__ inline const void* operator[](int i) const { return ins_addr[i]; }

  PointerToPointer() = default;
  PointerToPointer(const phi::GPUContext& ctx,
                   size_t in_num const T** pre_alloced_host_ptr) {
    auto* dev_ins_ptr = phi::memory_utils::Alloc(
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
    ins_addr = reinterpret_cast<void**>((*dev_ins_ptr)->ptr());
  }
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

// ConcatTensorWithDifferentShape<IndexT, MovSize, decltype(ptr_col_array)>
template <typename IndexT>
__global__ void ConcatCooSetIndicesKernel(const IndexT out_nnz,
                                          const IndexT* indice_offset,
                                          const IndexT* col_length,
                                          IndexT* output) {
  IndexT curr_segment = 0;

  // #define CUDA_KERNEL_LOOP_TYPE(i, num, index_type)
  // int64_t __index__ =
  //     static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // int64_t __stride__ = static_cast<int64_t>(blockDim.x) * gridDim.x;
  // for (index_type i = __index__; __index__ < (num);
  //      __index__ += __stride__, i = __index__)
  CUDA_KERNEL_LOOP_TYPE(tid_x, out_nnz, IndexT) {
    IndexT curr_col_offset = col_length[curr_segment + 1];
    // curr_offset 初始化到最接近tid的对一轮
    // 感觉这里的逻辑是让代码对应到最近tid那一段,毕竟每一轮tid都要递增
    while (curr_col_offset <= tid_x) {
      ++curr_segment;
      curr_col_offset = col_length[curr_segment + 1];
    }
    output[axis * out_nnz + tid_x] += indice_offset[curr_segment];
  }
}

template <typename IndexT, typename PointerWrapperT, typename DarrayWrapperT>
__global__ void ConcatCsr2D0ASetCrowsKernel(const IndexT total_crows_length,
                                            PointerWrapperT in_crows_data,
                                            DarrayWrapperT crows_length,
                                            IndexT* out_crows_data) {
  IndexT curr_segment = 0;

  CUDA_KERNEL_LOOP_TYPE(tid_x, total_crows_length, IndexT) {
    if (tid_x == 0) {
      out_crows_data[0] = 0;
    }
    // 优化
    IndexT curr_col_offset = crows_length[curr_segment + 1];
    IndexT curr_offset = crows_length[curr_segment];
    // curr_offset 初始化到最接近tid的对一轮
    // 感觉这里的逻辑是让代码对应到最近tid那一段,毕竟每一轮tid都要递增
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = crows_length[curr_segment + 1];
    }
    IndexT local_col = tid_x - curr_offset;
    // 注意这里tid_x对应的起始位置不包含crows的第0位,同理,在in_crows中也是如此
    output[tid_x + 1] =
        in_crows_data[curr_segment][local_col + 1] + crows_length[curr_segment];
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsrGet2DRowsNnzKernel(const IndexT total_rows,
                                            const IndexT rows,
                                            PointerWrapperT in_crows_data,
                                            IndexT* in_rows_nnzs) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_rows, IndexT) {
    // 优化
    IndexT curr_offset = tid_x / rows;
    IndexT index = tid_x % rows;
    in_rows_nnzs[tid] = in_crows_data[index][curr_offset + 1] -
                        in_crows_data[index][curr_offset];
  }
}

template <typename T,
          typename IndexT,
          typename PointerWrapperT,
          typename PointerWrapperIndexT,
          typename DarrayWrapperT>
__global__ void ConcatCsr2D1ASetValueKernel(const IndexT total_nnz,
                                            const size_t in_num,
                                            PointerWrapperT in_values_data,
                                            PointerWrapperIndexT in_cols_data,
                                            IndexT* in_rows_nnzs_data,
                                            IndexT* in_rows_index,
                                            DarrayWrapperT col_offsets,
                                            T* out_values,
                                            IndexT* out_cols, ) {
  IndexT curr_segment = 0;
  IndexT curr_col_offset = 0;
  IndexT curr_offset = 0;
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_nnz, IndexT) {
    // 优化
    // 这里in_rows_nnzs_data保存的是当前的每行的nnz个数,而不是递增的值
    curr_col_offset = curr_offset + in_rows_nnzs_data[curr_segment + 1];
    // curr_offset 初始化到最接近tid的对一轮
    // 感觉这里的逻辑是让代码对应到最近tid那一段,毕竟每一轮tid都要递增
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset += in_rows_nnzs_data[curr_segment + 1];
    }
    IndexT local_col = tid_x - curr_offset;
    Index total_offset = 0;
    Index index = curr_segment % rows;
    for (Index i = index; i <= curr_segment; i += in_num) {
      total_offset += in_rows_nnzs_data[i];
    }
    total_offset += local_col;
    out_values[tid] = in_values_data[index][total_offset];
    out_cols[tid] = in_cols_data[index][total_offset] + col_offsets[index];
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr2D1ASetCrowsKernel(const IndexT crows_nums,
                                            const size_t in_num,
                                            PointerWrapperT in_crows_data,
                                            IndexT* out_crows_data) {
  IndexT total_crows = 0;
  // 优化
  CUDA_KERNEL_LOOP_TYPE(tid_x, crows_nums, IndexT) {
    for (int i = 0; i != in_num; i++) {
      total_crows += in_crows_data[i][tid_x];
    }

    output[tid_x] = total_crows;
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsrGet3D1ACrowsKernel(
    const IndexT total_crows,
    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_crows_data,

    DarrayWrapperT in_rows,  // in_rows表示每一个的列数
    IndexT* in_matrix_nnx) {
  // 获取每轮(batch),也就是一个matrix下的nnz的数目
  // 对应于crows下每一个0开始到rows下的最后一位的数值
  // 例如 crows_data = [0,1,3,5, 8 , 0, 3, 4, 5 ,6] 这里两个需要获取的值是8和6.
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_crows, IndexT) {
    // total_crows == batch* in_num
    IndexT index = tid_x / batch;
    IndexT b = tid_x % batch;
    IndexT pos = b * (in_rows[index] + 1) - 1;
    in_matrix_nnx[tid] = in_crows_data[index][pos];
    // TODO(bapijun) 计算 in_matrix_nnx_offset
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D1ASetCrowsKernel(
    const IndexT out_crows_size,
    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_crows_data,
    DarrayWrapperT in_matrix_nnx_offset,  // 每个matrix中nnz的个数
    DarrayWrapperT in_rows_offsets,       //
    IndexT* out_crows) {
  // index表示位于第几个输入的
  IndexT index = 0;
  IndexT next_offset = 0;
  IndexT curr_offset = 0;
  // 每一轮batch中的0
  if (b == 0 && local_col == 0) {
    out_crows[in_nnz_offsets[b]] = 0;
  }
  CUDA_KERNEL_LOOP_TYPE(tid_x, out_crows_size, IndexT) {
    // out_crows_size == batch *
    // (各个in_num的rows的和)//注意这里第一位就让第一位处理即可
    IndexT b = tid_x % batch;
    IndexT rows_size = tid_x / batch;
    next_offset = in_rows_offsets[index + 1];
    curr_offset = in_rows_offsets[index];
    // curr_offset 初始化到最接近tid的对一轮
    // 感觉这里的逻辑是让代码对应到最近tid那一段,毕竟每一轮tid都要递增
    while (next_offset <= rows_size) {
      curr_offset = next_offset;
      ++index;
      next_offset = in_rows_offsets[index + 1];
    }
    // IndexT  rows = in_rows_offsets[index];
    IndexT local_col = rows_size - curr_offset;
    // 在 in_matrix_nnx_offset中位置p =  index * batch + b
    // out_crows中由于存在index个0
    out_crows[tid + index] = in_crows_data[index][local_col] +
                             in_matrix_nnx_offset[index * batch + b];
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D1ASetValuesColsKernel(
    const IndexT total_nnz,
    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_values_data,
    DarrayWrapperT
        in_matrix_nnx_offset,  // 每个matrix(也就是每轮batch)中nnz的累加值
    DarrayWrapperT in_batch_offsets,  // 这里是
    IndexT* out_values) {
  IndexT b = 0;
  IndexT next_offset = 0;
  IndexT curr_offset = 0;
  // total_nnz = in_matrix_nnx的和 =
  CUDA_KERNEL_LOOP_TYPE(tid_x, total_nnz, IndexT) {
    // in_batch_crows 表示每一轮
    next_offset += in_batch_crows[b + 1];
    curr_offset += in_batch_crows[b];
    // curr_offset 初始化到最接近tid的对一轮
    // 感觉这里的逻辑是让代码对应到最近tid那一段,毕竟每一轮tid都要递增
    while (next_offset <= tid_x) {
      curr_offset = next_offset;
      ++b;
      next_offset += in_batch_crows[b + 1];
    }
    IndexT local_col = tid_x - curr_offset;
    // p =  index * batch + b
    IndexT j = (b - 1) * in_num;

    next_offset = in_matrix_nnx_offset[j + 1];
    curr_offset = in_matrix_nnx_offset[j];

    while (next_offset <= local_col) {
      curr_offset = next_offset;
      ++j;
      next_offset = in_matrix_nnx_offset[j + 1];
    }
    IndexT index = j - (b - 1) * in_num;
    IndexT local_col2 = local_col - curr_offset;
    // local_col2 += index_offset[b];//每一轮k的叠加值
    for (int k = b; k > 0; k--) {
      local_col2 += in_crows_data[index][k * in_rows[index]];
    }

    out_values[tid_x] = in_values[index][local_col2];
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D2ASetCrowsKernel(
    const IndexT crows_num,
    const size_t in_num,
    PointerWrapperT in_crows_data,
    DarrayWrapperT
        in_matrix_nnx_offset,  // 每个matrix(也就是每轮batch)中nnz的累加值

    DarrayWrapperT in_batch_offsets,  // 这里是
    IndexT* out_crows) {
  // total_nnz = in_matrix_nnx的和 =
  CUDA_KERNEL_LOOP_TYPE(tid_x, crows_num, IndexT) {
    // TODO(bapijun) 使用什么方式优化
    IndexT total = 0;
    for (int i = 0; i < in_num; i++) {
      total = in_crows_data[i][tid_x];
    }
    out_crows[tid_x] = total;
  }
}

template <typename IndexT, typename PointerWrapperT>
__global__ void ConcatCsr3D2ASetvaluesKernel(
    const IndexT max_nnz,
    const size_t in_num,
    const IndexT batch,
    PointerWrapperT in_crows_data,
    DarrayWrapperT in_matrix_nnx_offset,  // 每个matrix(也就是每轮batch)中nnz的累加值
    PointerWrapperT in_max_batch_nnz,
    DarrayWrapperT in_batch_offsets,  // 这里是每一轮batch的最大值
    IndexT* out_values) {
  // total_nnz = in_matrix_nnx的和 =

  // 根据i和b获取的最大nnz 可以在上一个核函数获取
  IndexT b = 0;
  IndexT next_offset = 0;
  IndexT curr_offset = 0;
  IndexT now_nnz = in_max_batch_nnz[tid_z][tid_y];
  CUDA_KERNEL_LOOP_TYPE(tid_x, now_nnz, IndexT) {
    // TODO(bapijun) 使用什么方式优化
    next_offset += in_batch_crows[b + 1];
    curr_offset += in_batch_crows[b];
    // curr_offset 初始化到最接近tid的对一轮
    // 感觉这里的逻辑是让代码对应到最近tid那一段,毕竟每一轮tid都要递增
    while (next_offset <= tid_x) {
      curr_offset = next_offset;
      ++b;
      next_offset += in_batch_crows[b + 1];
    }
    IndexT local_col = tid_x - curr_offset;

    IndexT j = (b - 1) * in_num;
    next_offset = in_matrix_nnx_offset[j + in_num];
    curr_offset = in_matrix_nnx_offset[j];

    while (next_offset <= local_col) {
      curr_offset = next_offset;
      j = j+in_num;
      next_offset = in_matrix_nnx_offset[j + in_num];
    }
    
    // 根据in_batch_offsets 获取到当前的batch下的最大值
  }
}

template <typename T, typename IntT, typename Context>
void ConcatCooGPUKernel(const Context& dev_ctx,
                        const std::vector<const SparseCooTensor*>& x,
                        const Scalar& axis_scalar,
                        SparseCooTensor* out) {
  std::vector<DenseTensor> indices;
  std::vector<DenseTensor> values;
  std::vector<phi::DDim> x_dims;
  IntT in_num = ins.size();
  IntT axis = axis_scalar.to<IntT>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());
  IntT sparse_dim = x[0]->sparse_dim();
  IntT dense_dim = x[0]->dense_dim();

  DDim dims = x[0]->dims();
  DenseTensor out_indices;
  DenseTensor out_values;
  // 替换成 使用指针的形式
  funcs::ConcatFunctor<Context, T> concat_functor_value;
  funcs::ConcatFunctor<Context, IntT> concat_functor_indice;
  IntT pos = 0;
  for (const auto* t : x) {
    check_cat_sparse_dims(t, pos, dims, axis, sparse_dim, dense_dim);
    x_dims.push_back(t->dims());
    pos++;
  }
  // 迁移到对应的代码里面去,或者查看其他方式
  EmptyLikeCooKernel<T, Context>(dev_ctx, *x[0], out);
  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  if (axis < sparse_dim) {
    int64_t out_nnz = 0, out_cols = 0;
    std::vector<IndexT> indice_offsets(in_num, 0);
    std::vector<IndexT> in_cols(in_num + 1, 0);
    in_nnz[0] = 0;
    IntT i = 0;
    for (const auto* t : x) {
      indices.emplace_back(t->indices());
      values.emplace_back(t->values());
      out_nnz += t->nnz();
      in_cols[i + 1] = out_nnz;
      out_cols += t.dims()[axis];
      indice_offsets[i] = out_cols;
    }
    out_indices = phi::Empty<IndexT, Context>(dev_ctx, {sparse_dim, out_nnz});

    DDim v_dim = x[0]->values().dims();
    v_dim[0] = out_nnz;
    IntArray v_shape(v_dim.GetMutable(), v_dim.size());
    out_values = phi::Empty<T, Context>(dev_ctx, v_shape);

    // copy indice_offsets to device
    auto d_indice_offsets_tensor = memory_utils::Alloc(
        dev_ctx.GetPlace(),
        sizeof(IndexT) * indice_offsets.size(),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    IndexT* d_indice_offsets =
        reinterpret_cast<IndexT*>(d_indice_offsets_tensor->ptr());

    memory_utils::Copy(dev_ctx.GetPlace(),
                       d_indice_offsets,
                       phi::CPUPlace(),
                       indice_offsets.data(),
                       sizeof(int64_t) * indice_offsets.size(),
                       dev_ctx.stream());

    // copy in_cols to device
    auto d_in_cols_tensor = memory_utils::Alloc(
        dev_ctx.GetPlace(),
        sizeof(IndexT) * in_cols.size(),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    IndexT* d_in_cols = reinterpret_cast<IndexT*>(d_in_cols_tensor->ptr());

    memory_utils::Copy(dev_ctx.GetPlace(),
                       d_in_cols,
                       phi::CPUPlace(),
                       in_cols.data(),
                       sizeof(int64_t) * in_cols.size(),
                       dev_ctx.stream());

    // 因为在前面进行了检查,所以这个维度的nnz都一样
    concat_functor_indice(dev_ctx, indices, static_cast<int>(1), &out_indices);

    concat_functor_value(dev_ctx, values, static_cast<int>(0), &out_values);

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_nnz, 1);
    ConcatCooSetIndicesKernel<IntT><<<config.block_per_grid.x,
                                      config.thread_per_block.x,
                                      0,
                                      dev_ctx.stream()>>>(
        out_nnz,
        const IndexT* d_indice_offsets const IndexT* d_in_cols IndexT* output);

    out->SetMember(out_crows, out_cols, out_values, out_dims);
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
      auto concat_value =
          std::make_shared<DenseTensor>();  // 创建DenseTensor的智能指针
      concat_functor_value(dev_ctx, now_values, values_dim, concat_value.get());
      // 用 phi::funcs::StridedNumelCopyWithAxis<T, Context>
      values.push_back(*concat_value);
      indices.push_back(t->indices());
    }
    concat_functor_indice(dev_ctx, indices, static_cast<int>(1), &out_indices);
    concat_functor_value(dev_ctx, values, static_cast<int>(0), &out_values);

    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());
  }
}

template <typename T, typename Context>
void ConcatCooKernel(const Context& dev_ctx,
                     const std::vector<const SparseCooTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCooTensor* out) {
  int64_t out_nnz = 0;
  for (const auto* t : x) {
    out_nnz += t.nnz();
  }
  if (out_nnz < std::numeric_limits<int32_t>::max()) {
    ConcatCooGPUKernel<T, int32_t>(dev_ctx, x, axis_scalar, out);
  } else {
    ConcatCooGPUKernel<T, int64_t>(dev_ctx, x, axis_scalar, out);
  }
}

template <typename T, typename IntT, typename Context>
void ConcatCsrGPU2D0A(const Context& dev_ctx,
                      const std::vector<const SparseCooTensor*>& x,
                      size_t in_num,
                      const phi::DDim& out_dims,
                      T* out_values_data,
                      IntT* out_cols_data,
                      const DenseTensor& out_values,
                      const DenseTensor& out_cols,
                      const std::vector<const T*>& values_data_vec,
                      const std::vector<const int64_t*>& cols_data_vec,
                      const std::vector<const int64_t*>& crows_data_vec,
                      SparseCooTensor* out) {
  // 到这里为止所有的代码可以合并到前文去,这里留在这里只是为了未来方便调试

  // 除了第一个0 之外,按照row的次数叠加
  int64_t out_crows_size = 0;
  std::vector<int64_t> nnz_vec(in_num, 0);
  std::vector<int64_t> crows_length(in_num + 1, 0);
  for (size_t i = 0; i < in_num; i++) {
    nnz_vec.push_back(t->nnz());
    int64_t crows_num = static_cast<int64_t>(x[i]->dims()[0]) + 1;
    out_crows_size += crows_num;
    crows_length[i + 1] = out_crows_size;
  }
  out_crows_size++;
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  int64_t* out_crows_data = out_crows.data<int64_t>();

  auto gpu_place = dev_ctx.GetPlace();
  auto stream = dev_ctx.stream();
  int64_t value_offset = 0;
  // 用合并的concat方法替代
  for (size_t i = 0; i < in_num; i++) {
    int nnz = nnz_vec[i];
    // nnz == 0 的特殊情况,此时out_values_data指针很可能是错误的
    memory_utils::Copy(gpu_place,
                       out_values_data + value_offset,
                       gpu_place,
                       values_data_vec[i],
                       nnz * sizeof(T),
                       stream);
    memory_utils::Copy(gpu_place,
                       out_cols_data + value_offset,
                       gpu_place,
                       cols_data_vec[i],
                       nnz * sizeof(int64_t),
                       stream);
    value_offset += nnz;
  }
  const int64_t** crows_data_vec_data = crows_data_vec.data();

  PointerToPointer<int64_t> crows_ptr_array(
      dev_ctx, in_num, crows_data_vec_data);
  DArray<int64_t> d_crows_length(dev_ctx, crows_length);

  // 这里由于每一轮需要的只处理crow中0之后的部分 out_crows_size -1
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_crows_size - 1, 1);

  ConcatCsr2D0ASetCrowsKernel<int64_t,
                              decltype(crows_ptr_array),
                              decltype(d_crows_length)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(
          out_crows_size - 1, crows_ptr_array, d_crows_length, out_crows_data);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

void ConcatCsrGPU2D1A<T, IntT, Context>(
    const Context& dev_ctx,
    const std::vector<const SparseCooTensor*>& x,
    size_t in_num,
    const phi::DDim& out_dims,
    SparseCooTensor* out) {
  // 除了第一个0 之外,按照row的次数叠加
  std::vector<int64_t> col_offsets;

  int col_offset = 0 =, total_nnz = 0;
  col_offsets.push_back(col_offset);
  for (const auto* t : x) {
    col_offset += static_cast<int64_t>(x[i]->dims()[1]);
    col_offsets.push_back(col_offset);
    total_nnz += x[i]->nnz();
  }
  IntT rows = static_cast<size_t>(x[0]->dims()[0]);
  out_crows_size = rows + 1;
  int64_t total_rows = rows * in_num;

  DenseTensor out_crows = phi::Empty<IntT>(dev_ctx, {out_crows_size});
  IntT* out_crows_data = out_crows.data<IntT>();

  // 在设备中global memory的保存各个行的nnz个数
  DenseTensor in_rows_nnzs_tensor = phi::Empty<IntT>(dev_ctx, {total_rows});
  IntT* d_in_rows_nnzs_data = in_rows_nnzs_tensor.data<IntT>();

  PointerToPointer<IntT> crows_ptr_array(dev_ctx, in_num, crows_data_vec_data);

  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total_rows, 1);
  ConcatCsrGetRowsNnzKernel<IntT, decltype(crows_ptr_array)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(
          total_rows, rows, crows_ptr_array, d_in_rows_nnzs_data);

  PointerToPointer<T> values_ptr_array(dev_ctx, in_num, values_data_vec);
  PointerToPointer<IntT> cols_ptr_array(dev_ctx, in_num, cols_data_vec);
  DArray<IntT> d_col_offsets(dev_ctx, col_offsets);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total_nnz, 1);
  ConcatCsr2D1ASetValueKernel<T,
                              IntT,
                              decltype(values_ptr_array),
                              decltype(cols_ptr_array),
                              decltype(d_col_offsets)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(total_nnz,
                             in_num,
                             in_values_data,
                             in_cols_data,
                             d_in_rows_nnzs_data,
                             col_offsets,
                             out_values_data,
                             out_cols_data);

  config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, out_crows_size, 1);

  ConcatCsr2D1ASetCrowsKernel<IntT, decltype(crows_ptr_array)>
      <<<config.block_per_grid.x,
         config.thread_per_block.x,
         0,
         dev_ctx.stream()>>>(
          out_crows_size, in_num, crows_ptr_array, out_crows_data);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename IntT, typename Context>
void ConcatCsrGPU3D0A(const Context& dev_ctx,
                      const std::vector<const SparseCooTensor*>& x,
                      size_t in_num,
                      const phi::DDim& out_dims,
                      T* out_values_data,
                      IntT* out_cols_data,
                      const DenseTensor& out_values,
                      const DenseTensor& out_cols,
                      const std::vector<const T*>& values_data_vec,
                      const std::vector<const int64_t*>& cols_data_vec,
                      const std::vector<const int64_t*>& crows_data_vec,
                      SparseCooTensor* out) {
  // 到这里为止所有的代码可以合并到前文去,这里留在这里只是为了未来方便调试

  std::vector<DenseTensor> crows;
  std::vector<DenseTensor> values;
  std::vector<DenseTensor> cols;

  for (size_t i = 0; i < in_num; i++) {
    crows.emplace_back(x[i]->crows());
    values.emplace_back(x[i]->values());
    cols.emplace_back(x[i]->cols());
    out_crows_size += x[i]->crows().numel();
  }

  // axis==0 简单拼接所有的三个即可即可完成
  funcs::ConcatFunctor<Context, T> concat_functor;
  concat_functor(dev_ctx, values, static_cast<T>(0), &out_values);
  // cols的形状与value一致
  funcs::ConcatFunctor<Context, int64_t> concat_functor_indices;
  concat_functor_indices(dev_ctx, cols, static_cast<int64_t>(0), &out_cols);
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  concat_functor_indices(dev_ctx, crows, static_cast<int64_t>(0), &out_crows);

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

void ConcatCsrGPU3D1A<T, IntT, Context>(
    const Context& dev_ctx,
    const std::vector<const SparseCooTensor*>& x,
    size_t in_num,
    const phi::DDim& out_dims,
    T* out_values_data,
    IntT* out_cols_data,
    const DenseTensor& out_values,
    const DenseTensor& out_cols,
    const std::vector<const T*>& values_data_vec,
    const std::vector<const int64_t*>& cols_data_vec,
    const std::vector<const int64_t*>& crows_data_vec,
    SparseCooTensor* out) {
  // 除了第一个0 之外,按照row的次数叠加
  size_t batch = static_cast<int>(x[0]->dims()[0]);

  out_crows_size = batch;
  for (size_t i = 0; i < in_num; i++) {
    int64_t rows = static_cast<int64_t>(x[i]->dims()[1]);
    crows_numel.push_back(rows + 1);
    out_crows_size += batch * rows;
  }
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  int64_t* out_crows_data = out_crows.data<int64_t>();

  PointerToPointer<IntT> crows_ptr_array(dev_ctx, in_num, crows_data_vec_data);
  if (batch * in_num < 10) {  // 需要更精确的逻辑
    for (size_t b = 0; b < batch; b++) {
      // 针对每一轮batch的初始化
      out_crows_data[crow_index] = 0;
      crow_index++;
      cumulative_offset = 0;

      for (size_t i = 0; i < in_num; i++) {
        const int64_t* x_crows_ptr = x[i]->crows().data<int64_t>();
        // crows_numel[i] == 第i组的row+1
        int64_t x_crows_nnz = x_crows_ptr[(b + 1) * (crows_numel[i]) - 1];
        now_value_ptr = values_data_vec[i] + values_index[i];
        now_cols_ptr = cols_data_vec[i] + values_index[i];
        values_index[i] += x_crows_nnz;

        if (x_crows_nnz) {
          // nnz == 0 的特殊情况,此时out_values_data指针很可能是错误的
          memory_utils::Copy(cpu_place,
                             out_values_data + value_offset,
                             cpu_place,
                             now_value_ptr,
                             x_crows_nnz * sizeof(T));
          memory_utils::Copy(cpu_place,
                             out_cols_data + value_offset,
                             cpu_place,
                             now_cols_ptr,
                             x_crows_nnz * sizeof(int64_t));
        }

        value_offset += x_crows_nnz;
      }
    }
  } else {
  }

  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void ConcatCsrGPUKernel(const Context& dev_ctx,
                        const std::vector<const SparseCooTensor*>& x,
                        const Scalar& axis_scalar,
                        SparseCooTensor* out) {
  size_t in_num = x.size();

  int64_t axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());
  // 删掉
  std::vector<phi::DDim> x_dims;
  x_dims.reserve(in_num);
  std::vector<int64_t> crows_numel;
  std::vector<const int64_t*> crows_data_vec;
  std::vector<const T*> values_data_vec;
  std::vector<const int64_t*> cols_data_vec;
  crows_numel.reserve(in_num);
  crows_data_vec.reserve(in_num);
  values_data_vec.reserve(in_num);
  cols_data_vec.reserve(in_num);

  int64_t out_values_size = 0;
  int64_t out_crows_size = 0;
  for (const auto* t : x) {
    // TODO(bapijun) 考虑到nnz = 0的情况,进行补全`

    x_dims.emplace_back(t->dims());
    values_data_vec.push_back(t->values().data<T>());
    cols_data_vec.push_back(t->cols().data<int64_t>());
    // nnz == 0 时候,如果crow = [0] 这样的情况,补全0,避免之后的拼接遗漏
    crows_data_vec.push_back(t->crows().data<int64_t>());
    out_values_size += t->nnz();
  }
  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  T* out_values_data = out_values.data<T>();
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});
  int64_t* out_cols_data = out_cols.data<int64_t>();

  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  int x_dim = x_dims[0].size();
  if (x_dim == 2) {
    if (axis == 0) {
      ConcatCsrGPU2D0A<T, IntT, Context>(dev_ctx,
                                         x,
                                         out_dims,
                                         in_num,
                                         out_values_data,
                                         out_cols_data,
                                         crows_data_vec,
                                         out);
    } else {
      ConcatCsrGPU2D1A<T, IntT, Context>(dev_ctx,
                                         x,
                                         out_dims,
                                         in_num,
                                         out_values_data,
                                         out_cols_data,
                                         crows_data_vec,
                                         out);
    }

  } else if (x_dims.size() == 3) {
    // ConcatCsrGPU3D<T, IntT, Context>(
    //   dev_ctx, x,  axis_scalar, out);
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
  int64_t out_nnz = 0;
  for (const auto* t : x) {
    out_nnz += t.nnz();
  }
  if (out_nnz < std::numeric_limits<int32_t>::max()) {
    ConcatCsrGPUKernel<T, int32_t>(dev_ctx, x, axis_scalar, out);
  } else {
    ConcatCsrGPUKernel<T, int64_t>(dev_ctx, x, axis_scalar, out);
  }
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
