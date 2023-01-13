/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"

namespace phi {
namespace funcs {

static inline void GetBlockDims(const phi::GPUContext& context,
                                int64_t num_rows,
                                int64_t num_cols,
                                dim3* block_dims,
                                dim3* grid_dims) {
  // Set the thread block and grid according to CurrentDeviceId
  const int kThreadsPerBlock = 1024;
  int block_cols = kThreadsPerBlock;
  if (num_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
    block_cols = ((num_cols + 31) >> 5) << 5;
  }
  int block_rows = kThreadsPerBlock / block_cols;
  *block_dims = dim3(block_cols, block_rows, 1);

  constexpr int waves = 1;
  int max_threads = context.GetMaxPhysicalThreadCount() * waves;
  int64_t max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

  int grid_cols =
      std::min((num_cols + block_cols - 1) / block_cols, max_blocks);
  int grid_rows = std::min(max_blocks / grid_cols,
                           std::max(num_rows / block_rows, (int64_t)1));
  *grid_dims = dim3(grid_cols, grid_rows, 1);
}

template <typename T, int Size>
struct PointerWrapper {
 public:
  const void* ins_addr[Size];
  __device__ inline const void* operator[](int i) const { return ins_addr[i]; }

  PointerWrapper() {}
  PointerWrapper(const phi::GPUContext& ctx,
                 const std::vector<phi::DenseTensor>& ins,
                 const T** pre_alloced_host_ptr) {
    for (auto i = 0; i < ins.size(); ++i) {
      ins_addr[i] = ins[i].data();
    }
  }
};

template <typename T>
struct PointerToPointer {
 public:
  void** ins_addr{nullptr};
  __device__ inline const void* operator[](int i) const { return ins_addr[i]; }

  PointerToPointer() {}
  PointerToPointer(const phi::GPUContext& ctx,
                   const std::vector<phi::DenseTensor>& ins,
                   const T** pre_alloced_host_ptr,
                   paddle::memory::AllocationPtr* dev_ins_ptr) {
    auto in_num = ins.size();
    for (auto i = 0; i < in_num; ++i) {
      pre_alloced_host_ptr[i] = ins[i].data<T>();
    }
    *dev_ins_ptr = paddle::memory::Alloc(
        ctx.GetPlace(),
        in_num * sizeof(T*),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    auto* restored = phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
        pre_alloced_host_ptr, in_num);
    paddle::memory::Copy(ctx.GetPlace(),
                         (*dev_ins_ptr)->ptr(),
                         phi::CPUPlace(),
                         restored,
                         in_num * sizeof(T*),
                         ctx.stream());
    ins_addr = reinterpret_cast<void**>((*dev_ins_ptr)->ptr());
  }
};

template <typename T, typename IndexT, int Size>
struct PointerAndColWrapper {
 public:
  IndexT col_length[Size];
  PointerAndColWrapper(const phi::GPUContext& ctx,
                       const std::vector<phi::DenseTensor>& ins,
                       const IndexT& inputs_col_num,
                       const T** pre_alloced_host_ptr,
                       IndexT* inputs_col) {
    for (auto i = 0; i < inputs_col_num; ++i) {
      col_length[i] = inputs_col[i];
    }
    ins_ptr_wrapper = PointerWrapper<T, Size>(ctx, ins, pre_alloced_host_ptr);
  }

  __device__ inline const void* operator[](int i) const {
    return ins_ptr_wrapper[i];
  }

 private:
  PointerWrapper<T, Size> ins_ptr_wrapper;
};

template <typename T, typename IndexT>
struct PointerToPointerAndCol {
 public:
  IndexT* col_length{nullptr};
  PointerToPointerAndCol(const phi::GPUContext& ctx,
                         const std::vector<phi::DenseTensor>& ins,
                         const IndexT inputs_col_num,
                         const T** pre_alloced_host_ptr,
                         IndexT* inputs_col,
                         paddle::memory::AllocationPtr* dev_ins_ptr,
                         paddle::memory::AllocationPtr* dev_col_ptr) {
    *dev_col_ptr = paddle::memory::Alloc(
        ctx.GetPlace(),
        inputs_col_num * sizeof(IndexT),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    auto* restored = phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
        inputs_col, inputs_col_num);
    paddle::memory::Copy(ctx.GetPlace(),
                         (*dev_col_ptr)->ptr(),
                         phi::CPUPlace(),
                         restored,
                         inputs_col_num * sizeof(IndexT),
                         ctx.stream());
    col_length = static_cast<IndexT*>((*dev_col_ptr)->ptr());
    ins_ptr_wrapper =
        PointerToPointer<T>(ctx, ins, pre_alloced_host_ptr, dev_ins_ptr);
  }

  __device__ inline const void* operator[](int i) const {
    return ins_ptr_wrapper[i];
  }

 private:
  PointerToPointer<T> ins_ptr_wrapper;
};

template <int MovSize>
struct alignas(MovSize) Packed {
  __device__ Packed() {
    // do nothing
  }
  union {
    char buf[MovSize];
  };
};

template <typename IndexT, int MovSize, typename PointerAndColWrapperT>
__global__ void ConcatTensorWithDifferentShape(
    const PointerAndColWrapperT ins_datas,
    int col_size,
    const IndexT output_rows,
    const IndexT output_cols,
    void* output) {
  Packed<MovSize>* dst = reinterpret_cast<Packed<MovSize>*>(output);

  IndexT curr_segment = 0;
  IndexT curr_offset = ins_datas.col_length[0];

  CUDA_KERNEL_LOOP_TYPE(tid_x, output_cols, IndexT) {
    IndexT curr_col_offset = ins_datas.col_length[curr_segment + 1];

    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = ins_datas.col_length[curr_segment + 1];
    }

    IndexT local_col = tid_x - curr_offset;
    IndexT segment_width = curr_col_offset - curr_offset;

    const Packed<MovSize>* input_ptr =
        reinterpret_cast<const Packed<MovSize>*>(ins_datas[curr_segment]);

    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y) {
      dst[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * segment_width + local_col];
    }
  }
}

template <typename IndexT, int MovSize, typename PointerWrapperT>
__global__ void ConcatTensorWithSameShape(const PointerWrapperT ins_data,
                                          const IndexT fixed_in_col,
                                          const IndexT out_rows,
                                          const IndexT out_cols,
                                          void* output_data) {
  Packed<MovSize>* dst = reinterpret_cast<Packed<MovSize>*>(output_data);
  CUDA_KERNEL_LOOP_TYPE(tid_x, out_cols, IndexT) {
    IndexT split = tid_x / fixed_in_col;
    IndexT in_offset = tid_x - split * fixed_in_col;
    const Packed<MovSize>* input_ptr =
        reinterpret_cast<const Packed<MovSize>*>(ins_data[split]);
    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < out_rows; tid_y += blockDim.y * gridDim.y) {
      dst[tid_y * out_cols + tid_x] =
          input_ptr[tid_y * fixed_in_col + in_offset];
    }
  }
}

#define IMPL_CONCATE_CUDA_KERNEL_HELPER(func_impl, ...) \
  func_impl(4, ##__VA_ARGS__);                          \
  func_impl(8, ##__VA_ARGS__);                          \
  func_impl(16, ##__VA_ARGS__);                         \
  func_impl(32, ##__VA_ARGS__);                         \
  func_impl(64, ##__VA_ARGS__);                         \
  func_impl(128, ##__VA_ARGS__);

template <typename T, typename IndexT, int MovSize>
void DispatchConcatWithDifferentShapeKernelLimitNum(
    const phi::GPUContext& ctx,
    const std::vector<phi::DenseTensor>& ins,
    const IndexT inputs_col_num,
    const T** inputs_data,
    IndexT* inputs_col,
    const IndexT out_row,
    const IndexT out_col,
    phi::DenseTensor* output,
    const IndexT in_num,
    const IndexT limit_num) {
  dim3 block_dims;
  dim3 grid_dims;
  GetBlockDims(ctx, out_row, out_col, &block_dims, &grid_dims);

#define IMPL_COMPLEX_CONCAT_CUDA_KERNEL_CASE(size_, ...)    \
  case size_: {                                             \
    PointerAndColWrapper<T, IndexT, size_> ptr_col_array(   \
        ctx, ins, inputs_col_num, inputs_data, inputs_col); \
    __VA_ARGS__;                                            \
  } break;
  switch (phi::backends::gpu::RoundToNextHighPowOfTwo(limit_num, 4)) {
    IMPL_CONCATE_CUDA_KERNEL_HELPER(
        IMPL_COMPLEX_CONCAT_CUDA_KERNEL_CASE,
        ConcatTensorWithDifferentShape<IndexT, MovSize, decltype(ptr_col_array)>
        <<<grid_dims, block_dims, 0, ctx.stream()>>>(
            ptr_col_array, inputs_col_num, out_row, out_col, output->data()));
    default: {
      paddle::memory::AllocationPtr dev_ins_ptr{nullptr};
      paddle::memory::AllocationPtr dev_col_ptr{nullptr};
      PointerToPointerAndCol<T, IndexT> ptr_col_array(ctx,
                                                      ins,
                                                      inputs_col_num,
                                                      inputs_data,
                                                      inputs_col,
                                                      &dev_ins_ptr,
                                                      &dev_col_ptr);
      ConcatTensorWithDifferentShape<IndexT, MovSize, decltype(ptr_col_array)>
          <<<grid_dims, block_dims, 0, ctx.stream()>>>(
              ptr_col_array, inputs_col_num, out_row, out_col, output->data());
    }
  }
#undef IMPL_COMPLEX_CONCAT_CUDA_KERNEL_CASE
}

template <typename T, typename IndexT>
void DispatchConcatWithDifferentShapeMovsize(
    const phi::GPUContext& ctx,
    const std::vector<phi::DenseTensor>& ins,
    const IndexT inputs_col_num,
    const T** inputs_data,
    IndexT* inputs_col,
    const IndexT out_row,
    const IndexT out_col,
    phi::DenseTensor* output,
    const IndexT mov_size,
    const IndexT in_num,
    const IndexT limit_num) {
  if (mov_size == 16) {
    DispatchConcatWithDifferentShapeKernelLimitNum<T, IndexT, 16>(
        ctx,
        ins,
        inputs_col_num,
        inputs_data,
        inputs_col,
        out_row,
        out_col,
        output,
        in_num,
        limit_num);
  } else if (mov_size == 8) {
    DispatchConcatWithDifferentShapeKernelLimitNum<T, IndexT, 8>(ctx,
                                                                 ins,
                                                                 inputs_col_num,
                                                                 inputs_data,
                                                                 inputs_col,
                                                                 out_row,
                                                                 out_col,
                                                                 output,
                                                                 in_num,
                                                                 limit_num);
  } else if (mov_size == 4) {
    DispatchConcatWithDifferentShapeKernelLimitNum<T, IndexT, 4>(ctx,
                                                                 ins,
                                                                 inputs_col_num,
                                                                 inputs_data,
                                                                 inputs_col,
                                                                 out_row,
                                                                 out_col,
                                                                 output,
                                                                 in_num,
                                                                 limit_num);
  } else if (mov_size == 2) {
    DispatchConcatWithDifferentShapeKernelLimitNum<T, IndexT, 2>(ctx,
                                                                 ins,
                                                                 inputs_col_num,
                                                                 inputs_data,
                                                                 inputs_col,
                                                                 out_row,
                                                                 out_col,
                                                                 output,
                                                                 in_num,
                                                                 limit_num);
  } else {
    DispatchConcatWithDifferentShapeKernelLimitNum<T, IndexT, 1>(ctx,
                                                                 ins,
                                                                 inputs_col_num,
                                                                 inputs_data,
                                                                 inputs_col,
                                                                 out_row,
                                                                 out_col,
                                                                 output,
                                                                 in_num,
                                                                 limit_num);
  }
}

template <typename T, typename IndexT, int MovSize>
void DispatchConcatWithSameShapeKernelLimitNum(
    const phi::GPUContext& ctx,
    const std::vector<phi::DenseTensor>& ins,
    const T** inputs_data,
    IndexT in_col,
    const IndexT out_row,
    const IndexT out_col,
    phi::DenseTensor* output,
    const IndexT in_num,
    const IndexT limit_num) {
  dim3 block_dims;
  dim3 grid_dims;
  GetBlockDims(ctx, out_row, out_col, &block_dims, &grid_dims);

#define IMPL_CONCAT_CUDA_KERNEL_CASE(size_, ...)               \
  case size_: {                                                \
    PointerWrapper<T, size_> ptr_array(ctx, ins, inputs_data); \
    __VA_ARGS__;                                               \
  } break;

  switch (phi::backends::gpu::RoundToNextHighPowOfTwo(limit_num, 4)) {
    IMPL_CONCATE_CUDA_KERNEL_HELPER(
        IMPL_CONCAT_CUDA_KERNEL_CASE,
        ConcatTensorWithSameShape<IndexT, MovSize, decltype(ptr_array)>
        <<<grid_dims, block_dims, 0, ctx.stream()>>>(
            ptr_array, in_col, out_row, out_col, output->data()));
    default: {
      paddle::memory::AllocationPtr dev_ins_ptr{nullptr};
      PointerToPointer<T> ptr_array(ctx, ins, inputs_data, &dev_ins_ptr);
      ConcatTensorWithSameShape<IndexT, MovSize, decltype(ptr_array)>
          <<<grid_dims, block_dims, 0, ctx.stream()>>>(
              ptr_array, in_col, out_row, out_col, output->data());
    }
  }
#undef IMPL_CONCAT_CUDA_KERNEL_CASE
}

#undef IMPL_CONCATE_CUDA_KERNEL_HELPER

template <typename T, typename IndexT>
void DispatchConcatWithSameShapeMovsize(
    const phi::GPUContext& ctx,
    const std::vector<phi::DenseTensor>& ins,
    const T** inputs_data,
    IndexT in_col,
    const IndexT out_row,
    const IndexT out_col,
    phi::DenseTensor* output,
    const IndexT mov_size,
    const IndexT in_num,
    const IndexT limit_num) {
  if (mov_size == 16) {
    DispatchConcatWithSameShapeKernelLimitNum<T, IndexT, 16>(ctx,
                                                             ins,
                                                             inputs_data,
                                                             in_col,
                                                             out_row,
                                                             out_col,
                                                             output,
                                                             in_num,
                                                             limit_num);
  } else if (mov_size == 8) {
    DispatchConcatWithSameShapeKernelLimitNum<T, IndexT, 8>(ctx,
                                                            ins,
                                                            inputs_data,
                                                            in_col,
                                                            out_row,
                                                            out_col,
                                                            output,
                                                            in_num,
                                                            limit_num);
  } else if (mov_size == 4) {
    DispatchConcatWithSameShapeKernelLimitNum<T, IndexT, 4>(ctx,
                                                            ins,
                                                            inputs_data,
                                                            in_col,
                                                            out_row,
                                                            out_col,
                                                            output,
                                                            in_num,
                                                            limit_num);
  } else if (mov_size == 2) {
    DispatchConcatWithSameShapeKernelLimitNum<T, IndexT, 2>(ctx,
                                                            ins,
                                                            inputs_data,
                                                            in_col,
                                                            out_row,
                                                            out_col,
                                                            output,
                                                            in_num,
                                                            limit_num);
  } else {
    DispatchConcatWithSameShapeKernelLimitNum<T, IndexT, 1>(ctx,
                                                            ins,
                                                            inputs_data,
                                                            in_col,
                                                            out_row,
                                                            out_col,
                                                            output,
                                                            in_num,
                                                            limit_num);
  }
}

template <typename T, typename IndexT>
void DispatchConcatKernel(const phi::GPUContext& ctx,
                          const std::vector<phi::DenseTensor>& ins,
                          const IndexT inputs_col_num,
                          const T** inputs_data,
                          IndexT* inputs_col,
                          const IndexT out_row,
                          const IndexT out_col,
                          phi::DenseTensor* output,
                          const IndexT in_num,
                          const IndexT limit_num,
                          bool has_same_shape) {
  constexpr IndexT MaxVecSize = 16 / sizeof(T);
  bool find_vecsize_flag = false;
  IndexT dispatch_vec_size = 1;
  for (IndexT vec_size = MaxVecSize; vec_size > 0; vec_size /= 2) {
    for (IndexT idx = 0; idx < in_num + 1; idx++) {
      // Since input_cols[0] is 0, we need to jump.
      const IndexT input_col = inputs_col[idx + 1] - inputs_col[idx];
      if (input_col % vec_size == 0) {
        if (idx == in_num - 1) {
          find_vecsize_flag = true;
        }
      } else {
        break;
      }
    }
    if (find_vecsize_flag) {
      dispatch_vec_size = vec_size;
      break;
    }
  }

  const int64_t vectorized_out_col = out_col / dispatch_vec_size;
  for (IndexT idx = 0; idx < in_num + 1; idx++) {
    inputs_col[idx] /= dispatch_vec_size;
  }
  const IndexT mov_size = sizeof(T) * dispatch_vec_size;
  if (has_same_shape) {
    // In same shape situation, each input's col are equal, so here we select to
    // use inputs_col[1].
    DispatchConcatWithSameShapeMovsize<T, IndexT>(ctx,
                                                  ins,
                                                  inputs_data,
                                                  inputs_col[1],
                                                  out_row,
                                                  vectorized_out_col,
                                                  output,
                                                  mov_size,
                                                  in_num,
                                                  limit_num);
  } else {
    DispatchConcatWithDifferentShapeMovsize<T, IndexT>(ctx,
                                                       ins,
                                                       inputs_col_num,
                                                       inputs_data,
                                                       inputs_col,
                                                       out_row,
                                                       vectorized_out_col,
                                                       output,
                                                       mov_size,
                                                       in_num,
                                                       limit_num);
  }
}

template <typename T>
__global__ void SplitKernel_(const T* input_data,
                             const int64_t in_row,
                             const int64_t in_col,
                             const int64_t* out_cols,
                             int out_cols_size,
                             T** outputs_data) {
  int64_t curr_segment = 0;
  int64_t curr_offset = out_cols[0];
  CUDA_KERNEL_LOOP_TYPE(tid_x, in_col, int64_t) {
    int64_t curr_col_offset = out_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = out_cols[curr_segment + 1];
    }

    int64_t local_col = tid_x - curr_offset;
    int64_t segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs_data[curr_segment];
    if (output_ptr != nullptr) {
      int64_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * segment_width + local_col] =
            input_data[tid_y * in_col + tid_x];
    }
  }
}

template <typename T>
__device__ void SplitKernelDetail(const T* input_data,
                                  const int64_t in_row,
                                  const int64_t in_col,
                                  const int64_t fixed_out_col,
                                  T** outputs_data) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, in_col, int64_t) {
    int64_t split = tid_x / fixed_out_col;
    int64_t in_offset = tid_x - split * fixed_out_col;
    T* output_ptr = outputs_data[split];
    if (output_ptr != nullptr) {
      int64_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * fixed_out_col + in_offset] =
            input_data[tid_y * in_col + tid_x];
    }
  }
}

template <typename T>
__global__ void SplitKernel_(const T* input_data,
                             const int64_t in_row,
                             const int64_t in_col,
                             const int64_t fixed_out_col,
                             T** outputs_data) {
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

template <typename T>
__global__ void SplitKernel_(const T* input_data,
                             const int64_t in_row,
                             const int64_t in_col,
                             const int64_t fixed_out_col,
                             T* outputs_addr0,
                             T* outputs_addr1) {
  T* outputs_data[2];
  outputs_data[0] = outputs_addr0;
  outputs_data[1] = outputs_addr1;
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

template <typename T>
__global__ void SplitKernel_(const T* input_data,
                             const int64_t in_row,
                             const int64_t in_col,
                             const int64_t fixed_out_col,
                             T* outputs_addr0,
                             T* outputs_addr1,
                             T* outputs_addr2) {
  T* outputs_data[3];
  outputs_data[0] = outputs_addr0;
  outputs_data[1] = outputs_addr1;
  outputs_data[2] = outputs_addr2;
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

template <typename T>
__global__ void SplitKernel_(const T* input_data,
                             const int64_t in_row,
                             const int64_t in_col,
                             const int64_t fixed_out_col,
                             T* outputs_addr0,
                             T* outputs_addr1,
                             T* outputs_addr2,
                             T* outputs_addr3) {
  T* outputs_data[4];
  outputs_data[0] = outputs_addr0;
  outputs_data[1] = outputs_addr1;
  outputs_data[2] = outputs_addr2;
  outputs_data[3] = outputs_addr3;
  SplitKernelDetail<T>(input_data, in_row, in_col, fixed_out_col, outputs_data);
}

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T, typename IndexT>
void ConcatFunctorWithIndexType(const phi::GPUContext& ctx,
                                const std::vector<phi::DenseTensor>& ins,
                                int axis,
                                phi::DenseTensor* output) {
  // TODO(zcd): Add input data validity checking
  IndexT in_num = ins.size();
  IndexT in_row = 1;
  auto dim_0 = ins[0].dims();
  for (int i = 0; i < axis; ++i) {
    in_row *= dim_0[i];
  }
  IndexT in_col = ins[0].numel() / in_row;
  IndexT out_row = in_row, out_col = 0;

  IndexT inputs_col_num = in_num + 1;
  std::vector<const T*> inputs_data_vec(in_num, nullptr);
  std::vector<IndexT> inputs_col_vec(inputs_col_num, 0);
  const T** inputs_data = inputs_data_vec.data();
  IndexT* inputs_col = inputs_col_vec.data();
#ifdef PADDLE_WITH_HIP
  // TODO(chentianyu03): try to find a method to remove the Alloc function
  paddle::memory::AllocationPtr data_alloc = paddle::memory::Alloc(
      paddle::platform::CUDAPinnedPlace(), in_num * sizeof(T*));
  inputs_data = reinterpret_cast<const T**>(data_alloc->ptr());
  paddle::memory::AllocationPtr col_alloc = paddle::memory::Alloc(
      paddle::platform::CUDAPinnedPlace(), inputs_col_num * sizeof(IndexT));
  inputs_col = reinterpret_cast<IndexT*>(col_alloc->ptr());
#endif

  bool has_same_shape = true;
  for (int i = 0; i < in_num; ++i) {
    IndexT t_cols = ins[i].numel() / in_row;
    if (has_same_shape) {
      has_same_shape &= (t_cols == in_col);
    }
    out_col += t_cols;
    inputs_col[i + 1] = out_col;
  }
  IndexT limit_num = has_same_shape ? in_num : inputs_col_num;

  DispatchConcatKernel<T, IndexT>(ctx,
                                  ins,
                                  inputs_col_num,
                                  inputs_data,
                                  inputs_col,
                                  out_row,
                                  out_col,
                                  output,
                                  in_num,
                                  limit_num,
                                  has_same_shape);

#ifdef PADDLE_WITH_HIP
  // Prevent pinned memory from being covered and release the memory after
  // kernel launch of the stream is executed (reapply pinned memory next time)
  auto* data_alloc_released = data_alloc.release();
  auto* col_alloc_released = col_alloc.release();
  ctx.AddStreamCallback([data_alloc_released, col_alloc_released] {
    VLOG(4) << "Delete cuda pinned at " << data_alloc_released;
    VLOG(4) << "Delete cuda pinned at " << col_alloc_released;
    paddle::memory::allocation::Allocator::AllocationDeleter(
        data_alloc_released);
    paddle::memory::allocation::Allocator::AllocationDeleter(
        col_alloc_released);
  });
#endif
}

template <typename T>
struct ConcatFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    if (output->numel() < std::numeric_limits<int32_t>::max()) {
      ConcatFunctorWithIndexType<T, int32_t>(context, input, axis, output);
    } else {
      ConcatFunctorWithIndexType<T, int64_t>(context, input, axis, output);
    }
  }
};

template <typename T>
class SplitFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    // NOTE(zhiqiu): split a tensor of shape [0,3,4] at axis=1, result in 3
    // tensors of shape [0,1,4]
    if (input.numel() == 0) {
      return;
    }

    // TODO(zcd): Add input data validity checking
    int o_num = outputs->size();
    int64_t out_row = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      out_row *= dim_0[i];
    }

    int64_t out0_col = ref_inputs[0]->numel() / out_row;
    int64_t in_col = 0, in_row = out_row;
    bool has_same_shape = true;

    int outputs_cols_num = o_num + 1;
    std::vector<T*> outputs_data_vec(o_num);
    std::vector<int64_t> outputs_cols_vec(outputs_cols_num);
    T** outputs_data = outputs_data_vec.data();
    int64_t* outputs_cols = outputs_cols_vec.data();

// There are some differences between hip runtime and NV runtime.
// In NV, when the pageable memory data less than 64K is transferred from
// hosttodevice, it will be automatically asynchronous.
// However, only pinned memory in hip can copy asynchronously
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device
// 3.2.6.1. Concurrent Execution between Host and Device
// Memory copies from host to device of a memory block of 64 KB or less
#ifdef PADDLE_WITH_HIP
    paddle::memory::AllocationPtr data_alloc, cols_alloc;
    // TODO(chentianyu03): try to find a method to remove the Alloc function
    data_alloc = paddle::memory::Alloc(paddle::platform::CUDAPinnedPlace(),
                                       o_num * sizeof(T*));
    outputs_data = reinterpret_cast<T**>(data_alloc->ptr());
    // TODO(chentianyu03): try to find a method to remove the Alloc function
    cols_alloc = paddle::memory::Alloc(paddle::platform::CUDAPinnedPlace(),
                                       (outputs_cols_num) * sizeof(int64_t));
    outputs_cols = reinterpret_cast<int64_t*>(cols_alloc->ptr());
#endif

    outputs_cols[0] = 0;
    for (int i = 0; i < o_num; ++i) {
      int64_t t_col = ref_inputs.at(i)->numel() / out_row;
      if (has_same_shape) {
        if (t_col != out0_col) has_same_shape = false;
      }
      in_col += t_col;
      outputs_cols[i + 1] = in_col;
      if (outputs->at(i) != nullptr) {
        outputs_data[i] = outputs->at(i)->data<T>();
      } else {
        outputs_data[i] = nullptr;
      }
    }

    dim3 block_dims;
    dim3 grid_dims;
    GetBlockDims(context, out_row, in_col, &block_dims, &grid_dims);

    paddle::memory::allocation::AllocationPtr tmp_dev_outs_data;
    T** dev_out_gpu_data = nullptr;
    if (!has_same_shape || o_num < 2 || o_num > 4) {
      // TODO(chentianyu03): try to find a method to remove the Alloc function
      tmp_dev_outs_data = paddle::memory::Alloc(
          context.GetPlace(),
          o_num * sizeof(T*),
          phi::Stream(reinterpret_cast<phi::StreamId>(context.stream())));
      auto* restored = phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
          outputs_data, o_num);
      paddle::memory::Copy(context.GetPlace(),
                           tmp_dev_outs_data->ptr(),
                           phi::CPUPlace(),
                           restored,
                           o_num * sizeof(T*),
                           context.stream());
      dev_out_gpu_data = reinterpret_cast<T**>(tmp_dev_outs_data->ptr());
    }

    if (has_same_shape) {
      if (o_num == 2) {
        SplitKernel_<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(),
            in_row,
            in_col,
            out0_col,
            outputs_data[0],
            outputs_data[1]);
      } else if (o_num == 3) {
        SplitKernel_<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(),
            in_row,
            in_col,
            out0_col,
            outputs_data[0],
            outputs_data[1],
            outputs_data[2]);
      } else if (o_num == 4) {
        SplitKernel_<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(),
            in_row,
            in_col,
            out0_col,
            outputs_data[0],
            outputs_data[1],
            outputs_data[2],
            outputs_data[3]);
      } else {
        SplitKernel_<<<grid_dims, block_dims, 0, context.stream()>>>(
            input.data<T>(), in_row, in_col, out0_col, dev_out_gpu_data);
      }
    } else {
      auto tmp_dev_ins_col_data =
          // TODO(chentianyu03): try to find a method to remove the Alloc
          // function
          paddle::memory::Alloc(
              context.GetPlace(),
              outputs_cols_num * sizeof(int64_t),
              phi::Stream(reinterpret_cast<phi::StreamId>(context.stream())));
      auto* restored = phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
          outputs_cols, outputs_cols_num);
      paddle::memory::Copy(context.GetPlace(),
                           tmp_dev_ins_col_data->ptr(),
                           phi::CPUPlace(),
                           restored,
                           outputs_cols_num * sizeof(int64_t),
                           context.stream());
      int64_t* dev_outs_col_data =
          reinterpret_cast<int64_t*>(tmp_dev_ins_col_data->ptr());

      SplitKernel_<<<grid_dims, block_dims, 0, context.stream()>>>(
          input.data<T>(),
          in_row,
          in_col,
          dev_outs_col_data,
          static_cast<int>(outputs_cols_num),
          dev_out_gpu_data);
    }

#ifdef PADDLE_WITH_HIP
    // Prevent the pinned memory value from being covered and release the memory
    // after the launch kernel of the stream is executed (reapply pinned memory
    // next time)
    auto* data_alloc_released = data_alloc.release();
    auto* cols_alloc_released = cols_alloc.release();
    context.AddStreamCallback([data_alloc_released, cols_alloc_released] {
      paddle::memory::allocation::Allocator::AllocationDeleter(
          data_alloc_released);
      paddle::memory::allocation::Allocator::AllocationDeleter(
          cols_alloc_released);
    });
#endif
  }
};

#define DEFINE_FUNCTOR(type)                           \
  template class ConcatFunctor<phi::GPUContext, type>; \
  template class SplitFunctor<phi::GPUContext, type>

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace funcs
}  // namespace phi
