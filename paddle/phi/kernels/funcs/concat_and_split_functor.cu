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
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"

namespace phi {
namespace funcs {

template <typename T, int Size>
struct PointerWarpper {
 public:
  const T* data[Size];
  __device__ inline const T* operator[](int i) const { return data[i]; }

  PointerWarpper() {}
  PointerWarpper(const phi::GPUContext& ctx,
                 const std::vector<phi::DenseTensor>& ins,
                 const int& in_num) {
    for (auto i = 0; i < in_num; ++i) {
      data[i] = ins[i].data<T>();
    }
  }
};

template <typename T>
struct PointerWarpper<T, 0> {
 public:
  T** data{nullptr};
  __device__ inline const T* operator[](int i) const { return data[i]; }

  PointerWarpper() {}
  PointerWarpper(const phi::GPUContext& ctx,
                 const std::vector<phi::DenseTensor>& ins,
                 const int& in_num) {
    std::vector<const T*> inputs_data_vec(in_num);
    const T** inputs_data = inputs_data_vec.data();
    for (auto i = 0; i < in_num; ++i) {
      inputs_data[i] = ins[i].data<T>();
    }
    paddle::memory::allocation::AllocationPtr tmp_dev_ins_data;
    tmp_dev_ins_data = paddle::memory::Alloc(
        ctx.GetPlace(),
        in_num * sizeof(T*),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    auto* restored = paddle::platform::RestoreHostMemIfCapturingCUDAGraph(
        inputs_data, in_num);
    paddle::memory::Copy(ctx.GetPlace(),
                         tmp_dev_ins_data->ptr(),
                         paddle::platform::CPUPlace(),
                         restored,
                         in_num * sizeof(T*),
                         ctx.stream());
    data = reinterpret_cast<T**>(tmp_dev_ins_data->ptr());
  }
};

template <typename T, typename IndexT, int Size>
struct DataAndColWarpper {
 public:
  IndexT col_data[Size];
  DataAndColWarpper(const phi::GPUContext& ctx,
                    const std::vector<phi::DenseTensor>& ins,
                    const int& in_num,
                    const int& ins_col_num,
                    int64_t* ins_col) {
    for (auto i = 0; i < ins_col_num; ++i) {
      col_data[i] = static_cast<IndexT>(ins_col[i]);
    }
    data_warpper = PointerWarpper<T, Size>(ctx, ins, in_num);
  }

  __device__ inline const T* operator[](int i) const { return data_warpper[i]; }

 private:
  PointerWarpper<T, Size> data_warpper;
};

template <typename T, typename IndexT>
struct DataAndColWarpper<T, IndexT, 0> {
 public:
  IndexT* col_data;
  DataAndColWarpper(const phi::GPUContext& ctx,
                    const std::vector<phi::DenseTensor>& ins,
                    const int& in_num,
                    const int& ins_col_num,
                    int64_t* ins_col) {
    auto tmp_dev_ins_col_data = paddle::memory::Alloc(
        ctx.GetPlace(),
        ins_col_num * sizeof(IndexT),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    std::vector<IndexT> inputs_col_vec(ins_col_num);
    IndexT* in_col = inputs_col_vec.data();
    if (!std::is_same<IndexT, int64_t>::value) {
      for (auto i = 0; i < ins_col_num; ++i) {
        in_col[i] = ins_col[i];
      }
    } else {
      in_col = reinterpret_cast<IndexT*>(ins_col);
    }
    auto* restored = paddle::platform::RestoreHostMemIfCapturingCUDAGraph(
        in_col, ins_col_num);
    paddle::memory::Copy(ctx.GetPlace(),
                         tmp_dev_ins_col_data->ptr(),
                         paddle::platform::CPUPlace(),
                         restored,
                         ins_col_num * sizeof(IndexT),
                         ctx.stream());
    col_data = static_cast<IndexT*>(tmp_dev_ins_col_data->ptr());

    data_warpper = PointerWarpper<T, 0>(ctx, ins, in_num);
  }

  __device__ inline const T* operator[](int i) const { return data_warpper[i]; }

 private:
  PointerWarpper<T, 0> data_warpper;
};

template <typename T, typename IndexT, typename WarpperT>
__global__ void ConcatKernel_(WarpperT ins_datas,
                              int col_size,
                              const IndexT output_rows,
                              const IndexT output_cols,
                              T* output) {
  IndexT curr_segment = 0;
  IndexT curr_offset = ins_datas.col_data[0];
  CUDA_KERNEL_LOOP_TYPE(tid_x, output_cols, IndexT) {
    IndexT curr_col_offset = ins_datas.col_data[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = ins_datas.col_data[curr_segment + 1];
    }

    IndexT local_col = tid_x - curr_offset;
    IndexT segment_width = curr_col_offset - curr_offset;

    const T* input_ptr = ins_datas[curr_segment];
    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * segment_width + local_col];
  }
}

template <typename T, typename IndexT, typename WarpperT>
__global__ void ConcatKernel(WarpperT ins_data,
                             const IndexT fixed_in_col,
                             const IndexT out_rows,
                             const IndexT out_cols,
                             T* output_data) {
  CUDA_KERNEL_LOOP_TYPE(tid_x, out_cols, IndexT) {
    IndexT split = tid_x * 1.0 / fixed_in_col;
    IndexT in_offset = tid_x - split * fixed_in_col;
    const T* input_ptr = ins_data[split];
    IndexT tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < out_rows; tid_y += blockDim.y * gridDim.y) {
      output_data[tid_y * out_cols + tid_x] =
          input_ptr[tid_y * fixed_in_col + in_offset];
    }
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

  int max_threads = context.GetMaxPhysicalThreadCount();
  int64_t max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

  int grid_cols =
      std::min((num_cols + block_cols - 1) / block_cols, max_blocks);
  int grid_rows = std::min(max_blocks / grid_cols,
                           std::max(num_rows / block_rows, (int64_t)1));
  *grid_dims = dim3(grid_cols, grid_rows, 1);
}

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */

template <typename T>
struct ConcatFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    // TODO(zcd): Add input data validity checking
    int in_num = input.size();
    int64_t in_row = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      in_row *= dim_0[i];
    }
    int64_t out_row = in_row, out_col = 0;
    int ins_col_num = in_num + 1;
    std::vector<int64_t> inputs_col_vec(ins_col_num);
    int64_t* ins_col = inputs_col_vec.data();
// There are some differences between hip runtime and NV runtime.
// In NV, when the pageable memory data less than 64K is transferred from
// hosttodevice, it will be automatically asynchronous.
// However, only pinned memory in hip can copy asynchronously
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device
// 3.2.6.1. Concurrent Execution between Host and Device
// Memory copies from host to device of a memory block of 64 KB or less
#ifdef PADDLE_WITH_HIP
    std::vector<const T*> inputs_data_vec(in_num);
    const T** inputs_data = inputs_data_vec.data();
    paddle::memory::AllocationPtr data_alloc, col_alloc;
    // TODO(chentianyu03): try to find a method to remove the Alloc function
    data_alloc = paddle::memory::Alloc(paddle::platform::CUDAPinnedPlace(),
                                       in_num * sizeof(T*));
    inputs_data = reinterpret_cast<const T**>(data_alloc->ptr());
    // TODO(chentianyu03): try to find a method to remove the Alloc function
    col_alloc = paddle::memory::Alloc(paddle::platform::CUDAPinnedPlace(),
                                      ins_col_num * sizeof(int));
    ins_col = reinterpret_cast<int64_t*>(col_alloc->ptr());
#endif

    ins_col[0] = 0;
    bool has_same_shape = true;
    int64_t in_col = input[0].numel() / in_row;
    for (int i = 0; i < in_num; ++i) {
      int64_t t_cols = input[i].numel() / in_row;
      if (has_same_shape) {
        has_same_shape &= (t_cols == in_col);
      }
      out_col += t_cols;
      ins_col[i + 1] = out_col;
    }

    dim3 block_dims;
    dim3 grid_dims;
    GetBlockDims(context, out_row, out_col, &block_dims, &grid_dims);

    bool use_int32 = output->numel() < std::numeric_limits<int32_t>::max();
    if (has_same_shape) {
#define IMPL_CONCAT_WITH_WARPPER(size, index_t)              \
  PointerWarpper<T, size> ptr_array(context, input, in_num); \
  ConcatKernel<T, index_t, decltype(ptr_array)>              \
      <<<grid_dims, block_dims, 0, context.stream()>>>(      \
          ptr_array, in_col, out_row, out_col, output->data<T>());
      if (use_int32) {
        if (in_num < 32) {
          IMPL_CONCAT_WITH_WARPPER(32, int32_t);
        } else if (in_num < 64) {
          IMPL_CONCAT_WITH_WARPPER(64, int32_t);
        } else if (in_num < 128) {
          IMPL_CONCAT_WITH_WARPPER(128, int32_t);
        } else {
          IMPL_CONCAT_WITH_WARPPER(0, int32_t);
        }
      } else {
        if (in_num < 32) {
          IMPL_CONCAT_WITH_WARPPER(32, int64_t);
        } else if (in_num < 64) {
          IMPL_CONCAT_WITH_WARPPER(64, int64_t);
        } else if (in_num < 128) {
          IMPL_CONCAT_WITH_WARPPER(128, int64_t);
        } else {
          IMPL_CONCAT_WITH_WARPPER(0, int64_t);
        }
      }
#undef IMPL_CONCAT_WITH_WARPPER
    } else {
#define IMPL_CONCAT_WITH_COMPLEX_WARPPER(size, index_t) \
  DataAndColWarpper<T, index_t, size> ptr_col_array(    \
      context, input, in_num, ins_col_num, ins_col);    \
  ConcatKernel_<T, index_t, decltype(ptr_col_array)>    \
      <<<grid_dims, block_dims, 0, context.stream()>>>( \
          ptr_col_array,                                \
          ins_col_num,                                  \
          static_cast<index_t>(out_row),                \
          static_cast<index_t>(out_col),                \
          output->data<T>());

      if (use_int32) {
        if (in_num < 32) {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(32, int32_t);
        } else if (in_num < 64) {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(64, int32_t);
        } else if (in_num < 128) {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(64, int32_t);
        } else {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(0, int32_t);
        }
      } else {
        if (in_num < 32) {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(32, int64_t);
        } else if (in_num < 64) {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(64, int64_t);
        } else if (in_num < 128) {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(64, int64_t);
        } else {
          IMPL_CONCAT_WITH_COMPLEX_WARPPER(0, int64_t);
        }
      }
#undef IMPL_CONCAT_WITH_COMPLEX_WARPPER
    }

#ifdef PADDLE_WITH_HIP
    // Prevent the pinned memory value from being covered and release the memory
    // after the launch kernel of the stream is executed (reapply pinned memory
    // next time)
    auto* data_alloc_released = data_alloc.release();
    auto* col_alloc_released = col_alloc.release();
    context.AddStreamCallback([data_alloc_released, col_alloc_released] {
      VLOG(4) << "Delete cuda pinned at " << data_alloc_released;
      VLOG(4) << "Delete cuda pinned at " << col_alloc_released;
      paddle::memory::allocation::Allocator::AllocationDeleter(
          data_alloc_released);
      paddle::memory::allocation::Allocator::AllocationDeleter(
          col_alloc_released);
    });
#endif
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
      auto* restored = paddle::platform::RestoreHostMemIfCapturingCUDAGraph(
          outputs_data, o_num);
      paddle::memory::Copy(context.GetPlace(),
                           tmp_dev_outs_data->ptr(),
                           paddle::platform::CPUPlace(),
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
      auto* restored = paddle::platform::RestoreHostMemIfCapturingCUDAGraph(
          outputs_cols, outputs_cols_num);
      paddle::memory::Copy(context.GetPlace(),
                           tmp_dev_ins_col_data->ptr(),
                           paddle::platform::CPUPlace(),
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
