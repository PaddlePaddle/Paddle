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

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/segmented_array.h"

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

#ifndef PADDLE_WITH_HIP
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x)
#endif
#else
#define PADDLE_ALIGN(x)
#endif

template <typename T, int Size>
struct PointerWrapper {
 public:
  const void* ins_addr[Size];
  __device__ inline const void* operator[](int i) const { return ins_addr[i]; }

  PointerWrapper() = default;
  PointerWrapper(const phi::GPUContext& ctx,
                 const std::vector<phi::DenseTensor>& ins,
                 const T** pre_alloced_host_ptr) {
    SetInputAddr(ins);
  }

 protected:
  void SetInputAddr(const std::vector<phi::DenseTensor>& ins) {
    for (auto i = 0; i < ins.size(); ++i) {
      ins_addr[i] = ins[i].data();
    }
  }
};

template <typename T, int Size>
struct PADDLE_ALIGN(256) AlignedPointerWrapper
    : public PointerWrapper<T, Size> {
 public:
  AlignedPointerWrapper() = default;
  AlignedPointerWrapper(const phi::GPUContext& ctx,
                        const std::vector<phi::DenseTensor>& ins,
                        const T** pre_alloced_host_ptr) {
    this->SetInputAddr(ins);
  }
};

template <typename T>
struct PointerToPointer {
 public:
  void** ins_addr{nullptr};
  __device__ inline const void* operator[](int i) const { return ins_addr[i]; }

  PointerToPointer() = default;
  PointerToPointer(const phi::GPUContext& ctx,
                   const std::vector<phi::DenseTensor>& ins,
                   const T** pre_alloced_host_ptr,
                   phi::Allocator::AllocationPtr* dev_ins_ptr) {
    auto in_num = ins.size();
    for (auto i = 0; i < in_num; ++i) {
      pre_alloced_host_ptr[i] = ins[i].data<T>();
    }
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
    ins_addr = reinterpret_cast<void**>((*dev_ins_ptr)->ptr());
  }
};

template <typename T, typename IndexT, int Size>
struct PADDLE_ALIGN(256) PointerAndColWrapper {
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
                         phi::Allocator::AllocationPtr* dev_ins_ptr,
                         phi::Allocator::AllocationPtr* dev_col_ptr) {
    *dev_col_ptr = phi::memory_utils::Alloc(
        ctx.GetPlace(),
        inputs_col_num * sizeof(IndexT),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    auto* restored = phi::backends::gpu::RestoreHostMemIfCapturingCUDAGraph(
        inputs_col, inputs_col_num);
    memory_utils::Copy(ctx.GetPlace(),
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

#undef PADDLE_ALIGN

template <int MovSize>
struct alignas(MovSize) Packed {
  __device__ Packed() = default;
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
      phi::Allocator::AllocationPtr dev_ins_ptr{nullptr};
      phi::Allocator::AllocationPtr dev_col_ptr{nullptr};
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

#define IMPL_CONCAT_CUDA_KERNEL_CASE(size_, ...)                      \
  case size_: {                                                       \
    AlignedPointerWrapper<T, size_> ptr_array(ctx, ins, inputs_data); \
    __VA_ARGS__;                                                      \
  } break;

  switch (phi::backends::gpu::RoundToNextHighPowOfTwo(limit_num, 4)) {
    IMPL_CONCATE_CUDA_KERNEL_HELPER(
        IMPL_CONCAT_CUDA_KERNEL_CASE,
        ConcatTensorWithSameShape<IndexT, MovSize, decltype(ptr_array)>
        <<<grid_dims, block_dims, 0, ctx.stream()>>>(
            ptr_array, in_col, out_row, out_col, output->data()));
    default: {
      phi::Allocator::AllocationPtr dev_ins_ptr{nullptr};
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

  auto output_data = reinterpret_cast<std::uintptr_t>(output->data());
  for (IndexT vec_size = MaxVecSize; vec_size > 0; vec_size /= 2) {
    const IndexT mov_size = vec_size * sizeof(T);
    for (IndexT idx = 1; idx < in_num + 1; idx++) {
      auto input_data = reinterpret_cast<std::uintptr_t>(inputs_data[idx - 1]);
      // Since input_cols[0] is 0, we need to jump.
      const IndexT input_col = inputs_col[idx] - inputs_col[idx - 1];
      if (input_col % vec_size == 0 && output_data % mov_size == 0 &&
          input_data % mov_size == 0) {
        if (idx == in_num) {
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
  for (size_t i = 0; i < ins.size(); ++i) {
    inputs_data_vec[i] = ins[i].data<T>();
  }
  std::vector<IndexT> inputs_col_vec(inputs_col_num, 0);
  const T** inputs_data = inputs_data_vec.data();
  IndexT* inputs_col = inputs_col_vec.data();

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

template <typename T, typename IndexT, funcs::SegmentedArraySize Size>
struct PointerAndColArray
    : public funcs::PointerArraySetter<phi::GPUContext, T, Size> {
 public:
  funcs::ValueArray<IndexT, Size> val_array;

  PointerAndColArray() = default;
  PointerAndColArray(const phi::GPUContext& ctx,
                     const int out_col_num,
                     IndexT* out_cols,
                     std::vector<DenseTensor*>* t,
                     T** pre_alloc_host_buf = nullptr)
      : funcs::PointerArraySetter<phi::GPUContext, T, Size>(
            ctx,
            t,
            /*need_alloc=*/false,
            /*use_cuda_graph=*/true,
            pre_alloc_host_buf) {
    IndexT* dev_ptr = nullptr;
    if (Size == SegmentedArraySize::kVariableLength) {
      size_t num_bytes = out_col_num * sizeof(IndexT);
      dev_ptr = reinterpret_cast<IndexT*>(this->AllocAndCopy(
          ctx, reinterpret_cast<void*>(out_cols), num_bytes, true));
      val_array.Set(dev_ptr, out_col_num);
    } else {
      val_array.Set(out_cols, out_col_num);
    }
  }
};

template <typename T, typename IndexT, typename DataArrayT>
__global__ void SplitTensorWithSameShape(const T* input_data,
                                         const IndexT out_row,
                                         const IndexT cumulative_col,
                                         const IndexT fixed_out_col,
                                         DataArrayT data_array) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, cumulative_col, IndexT) {
    IndexT split = tid_x / fixed_out_col;
    IndexT in_offset = tid_x - split * fixed_out_col;
    T* output_ptr = data_array.data[split];
    if (output_ptr != nullptr) {
      IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < out_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * fixed_out_col + in_offset] =
            input_data[tid_y * cumulative_col + tid_x];
    }
  }
}

template <typename T, typename IndexT, typename DataArrayT, typename ValArrayT>
__global__ void SplitTensorWithDifferentShape(const T* input_data,
                                              const IndexT out_row,
                                              const IndexT cumulative_col,
                                              DataArrayT data_array,
                                              ValArrayT col_array) {
  IndexT curr_segment = 0;
  IndexT curr_offset = col_array.data[0];
  CUDA_KERNEL_LOOP_TYPE(tid_x, cumulative_col, IndexT) {
    IndexT curr_col_offset = col_array.data[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = col_array.data[curr_segment + 1];
    }

    IndexT local_col = tid_x - curr_offset;
    IndexT segment_width = curr_col_offset - curr_offset;
    T* output_ptr = data_array.data[curr_segment];
    if (output_ptr != nullptr) {
      IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < out_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * segment_width + local_col] =
            input_data[tid_y * cumulative_col + tid_x];
    }
  }
}

template <typename T, typename IndexT, funcs::SegmentedArraySize Size>
void SplitFunctionDispatchWithSameShape(const phi::GPUContext& ctx,
                                        const IndexT out_col,
                                        const IndexT out_row,
                                        const IndexT cumulative_col,
                                        const T* input_data,
                                        std::vector<phi::DenseTensor*>* outs,
                                        T** pre_alloc_host_buf) {
  dim3 grid_dims;
  dim3 block_dims;
  GetBlockDims(ctx, out_row, cumulative_col, &block_dims, &grid_dims);

  funcs::PointerArraySetter<phi::GPUContext, T, Size> setter(
      ctx,
      outs,
      /*need_alloc=*/false,
      /*use_cuda_graph=*/true,
      pre_alloc_host_buf);
  SplitTensorWithSameShape<T, IndexT, decltype(setter.array)>
      <<<grid_dims, block_dims, 0, ctx.stream()>>>(
          input_data, out_row, cumulative_col, out_col, setter.array);
}

template <typename T, typename IndexT, funcs::SegmentedArraySize Size>
void SplitFunctionDispatchWithDifferentShape(
    const phi::GPUContext& ctx,
    const int out_col_num,
    const IndexT out_row,
    const IndexT cumulative_col,
    const T* input_data,
    std::vector<phi::DenseTensor*>* outs,
    IndexT* output_cols,
    T** pre_alloc_host_buf) {
  dim3 grid_dims;
  dim3 block_dims;
  GetBlockDims(ctx, out_row, cumulative_col, &block_dims, &grid_dims);
  PointerAndColArray<T, IndexT, Size> setter(
      ctx, out_col_num, output_cols, outs, pre_alloc_host_buf);

  SplitTensorWithDifferentShape<T,
                                IndexT,
                                decltype(setter.array),
                                decltype(setter.val_array)>
      <<<grid_dims, block_dims, 0, ctx.stream()>>>(
          input_data, out_row, cumulative_col, setter.array, setter.val_array);
}

template <typename T, typename IndexT>
void SplitFunctorDispatchWithIndexType(
    const phi::GPUContext& ctx,
    int axis,
    const phi::DenseTensor& input,
    const std::vector<const phi::DenseTensor*>& ref_ins,
    std::vector<phi::DenseTensor*>* outs) {
  // TODO(zcd): Add input data validity checking
  int out_num = outs->size();
  IndexT out_row = 1;
  auto ref_dim = ref_ins[0]->dims();
  for (int i = 0; i < axis; ++i) {
    out_row *= ref_dim[i];
  }
  IndexT out_col = ref_ins[0]->numel() / out_row;
  IndexT cumulative_col = 0;
  bool has_same_shape = true;

  int out_cols_num = out_num + 1;
  std::vector<IndexT> outputs_cols_vec(out_cols_num, 0);
  IndexT* outs_cols = outputs_cols_vec.data();
  T** outs_data = nullptr;

  outs_cols[0] = 0;
  for (int i = 0; i < out_num; ++i) {
    IndexT t_col = ref_ins.at(i)->numel() / out_row;
    if (has_same_shape) {
      has_same_shape &= (t_col == cumulative_col);
    }
    cumulative_col += t_col;
    outs_cols[i + 1] = cumulative_col;
  }
  int limit_num = has_same_shape ? out_num : out_cols_num;
  if (has_same_shape) {
    switch (funcs::CalcArraySize(limit_num)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          SplitFunctionDispatchWithSameShape<T, IndexT, kArraySize>(
              ctx,
              out_col,
              out_row,
              cumulative_col,
              input.data<T>(),
              outs,
              outs_data));
    }
  } else {
    switch (funcs::CalcArraySize(limit_num)) {
      SEGMENTED_ARRAY_KERNEL_HELPER(
          SplitFunctionDispatchWithDifferentShape<T, IndexT, kArraySize>(
              ctx,
              out_cols_num,
              out_row,
              cumulative_col,
              input.data<T>(),
              outs,
              outs_cols,
              outs_data));
    }
  }
}

template <typename T>
class SplitFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    int64_t numel = input.numel();
    // NOTE(zhiqiu): split a tensor of shape [0,3,4] at axis=1, result in
    // 3 tensors of shape [0,1,4]
    if (input.numel() == 0) {
      return;
    }

    if (numel < std::numeric_limits<int32_t>::max()) {
      SplitFunctorDispatchWithIndexType<T, int32_t>(
          context, axis, input, ref_inputs, outputs);
    } else {
      SplitFunctorDispatchWithIndexType<T, int64_t>(
          context, axis, input, ref_inputs, outputs);
    }
  }
};

#define DEFINE_FUNCTOR(type)                           \
  template class ConcatFunctor<phi::GPUContext, type>; \
  template class SplitFunctor<phi::GPUContext, type>

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace funcs
}  // namespace phi
