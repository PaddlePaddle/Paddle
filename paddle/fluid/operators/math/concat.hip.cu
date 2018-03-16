/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hip/hip_runtime.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/operators/math/concat.h"
#include "paddle/fluid/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__device__ T upper_bound(const T* first, T count, T val) {
  const T* orig = first;
  const T* it = nullptr;
  T step = 0;
  while (count > 0) {
    it = first;
    step = count / 2;
    it += step;
    if (!(val < *it)) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first - orig;
}

template <typename T>
__global__ void KernelConcat(T** inputs, const int* input_cols, int col_size,
                             const int output_rows, const int output_cols,
                             T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int segment = upper_bound<int>(input_cols, col_size, tid_x) - 1;

  int curr_offset = input_cols[segment];
  int curr_segment = segment;
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    T curr_col_offset;
    while ((curr_col_offset = input_cols[curr_segment + 1]) <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* input_ptr = inputs[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * segment_width + local_col];
  }
}

template <typename T>
__global__ void KernelConcat(T** inputs, const int input_col,
                             const int output_rows, const int output_cols,
                             T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  double inv_input_col = 1.0 / input_col;
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * inv_input_col;
    int in_offset = tid_x - split * input_col;
    T* input_ptr = inputs[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y) {
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * input_col + in_offset];
    }
  }
}

template <typename T>
__global__ void KernelConcatGrad(const T* input, const int input_row,
                                 const int input_col, const int* output_cols,
                                 int col_size, T** outputs) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int segment = upper_bound<int>(output_cols, col_size, tid_x) - 1;
  int curr_offset = output_cols[segment];
  int curr_segment = segment;
  for (; tid_x < input_col; tid_x += blockDim.x * gridDim.x) {
    T curr_col_offset;
    while ((curr_col_offset = output_cols[curr_segment + 1]) <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < input_row; tid_y += blockDim.y * gridDim.y)
      output_ptr[tid_y * segment_width + local_col] =
          input[tid_y * input_col + tid_x];
  }
}

template <typename T>
__global__ void KernelConcatGrad(const T* input, const int input_row,
                                 const int input_col, const int output_cols,
                                 T** outputs) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  double inv_input_col = 1.0 / input_col;
  for (; tid_x < input_col; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * inv_input_col;
    int in_offset = tid_x - split * input_col;
    T* output_ptr = outputs[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < input_row; tid_y += blockDim.y * gridDim.y)
      output_ptr[tid_y * output_cols + in_offset] =
          input[tid_y * input_col + tid_x];
  }
}

/*
 * All tensors' dimension should be the same and the values of
 * each dimension are the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::vector<framework::Tensor>& input, const int axis,
                  framework::Tensor* output) {
    // TODO(zcd): Add input data validity checking
    int num = input.size();
    int rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int cols = input[0].numel() / rows;
    int out_rows = rows, out_cols = 0;

    framework::Vector<int16_t> inputs_data(num * sizeof(T*) / 2);
    framework::Vector<int> inputs_cols(num + 1);
    inputs_cols[0] = 0;
    T** inputs_ptr = reinterpret_cast<T**>(inputs_data.data());

    bool sameShape = true;
    for (int i = 0; i < num; ++i) {
      int t_cols = input[i].numel() / rows;
      if (sameShape) {
        if (t_cols != cols) sameShape = false;
      }
      out_cols += t_cols;
      inputs_cols[i + 1] = out_cols;
      inputs_ptr[i] = const_cast<T*>(input[i].data<T>());
    }

    T** ins_gpu =
        reinterpret_cast<T**>(inputs_data.CUDAMutableData(context.GetPlace()));
    const int* ins_col_gpu = inputs_cols.CUDAData(context.GetPlace());

    // computation
    // set the thread block and grid according to CurrentDeviceId
    const int kThreadsPerBlock = 1024;
    int block_cols = kThreadsPerBlock;
    if (out_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
      block_cols = ((out_cols + 31) >> 5) << 5;
    }
    int block_rows = kThreadsPerBlock / block_cols;
    dim3 block_size = dim3(block_cols, block_rows, 1);

    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    int grid_cols =
        std::min((out_cols + block_cols - 1) / block_cols, max_blocks);
    int grid_rows =
        std::min(max_blocks / grid_cols, std::max(out_rows / block_rows, 1));
    dim3 grid_size = dim3(grid_cols, grid_rows, 1);

    if (sameShape) {
      hipLaunchKernelGGL((KernelConcat<T>), dim3(grid_size), dim3(block_size), 0, context.stream(),
          ins_gpu, cols, out_rows, out_cols, output->data<T>());
    } else {
      hipLaunchKernelGGL((KernelConcat<T>), dim3(grid_size), dim3(block_size), 0, context.stream(),
          ins_gpu, ins_col_gpu, static_cast<int>(inputs_cols.size()), out_rows,
          out_cols, output->data<T>());
    }
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension are the same, except the axis dimension.
 */
template <typename T>
class ConcatGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const int axis,
                  std::vector<framework::Tensor>& outputs) {
    // TODO(zcd): Add input data validity checking
    int num = outputs.size();
    int input_row = 1;
    auto dim_0 = outputs[0].dims();
    for (int i = 0; i < axis; ++i) {
      input_row *= dim_0[i];
    }

    int output_col_0 = outputs[0].numel() / input_row;
    int input_col = 0;
    bool sameShape = true;

    framework::Vector<int16_t> outputs_data(num * sizeof(T*) / 2);
    framework::Vector<int> outputs_cols(num + 1);
    outputs_cols[0] = 0;
    T** outputs_ptr = reinterpret_cast<T**>(outputs_data.data());

    for (int i = 0; i < num; ++i) {
      int t_col = outputs[i].numel() / input_row;
      if (sameShape) {
        if (t_col != output_col_0) sameShape = false;
      }
      input_col += t_col;
      outputs_cols[i + 1] = input_col;
      outputs_ptr[i] = outputs[i].data<T>();
    }

    T** outs_gpu =
        reinterpret_cast<T**>(outputs_data.CUDAMutableData(context.GetPlace()));
    const int* outs_col_gpu = outputs_cols.CUDAData(context.GetPlace());

    // computation
    const int kThreadsPerBlock = 1024;
    int block_cols = kThreadsPerBlock;
    if (input_col < kThreadsPerBlock) {  // block_cols is aligned by 32.
      block_cols = ((input_col + 31) >> 5) << 5;
    }
    int block_rows = kThreadsPerBlock / block_cols;
    dim3 block_size = dim3(block_cols, block_rows, 1);

    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    int grid_cols =
        std::min((input_col + block_cols - 1) / block_cols, max_blocks);
    int grid_rows =
        std::min(max_blocks / grid_cols, std::max(input_row / block_rows, 1));
    dim3 grid_size = dim3(grid_cols, grid_rows, 1);

    if (sameShape) {
      hipLaunchKernelGGL((KernelConcatGrad<T>), dim3(grid_size), dim3(block_size), 0, context.stream(),
          input.data<T>(), input_row, input_col, output_col_0, outs_gpu);
    } else {
      hipLaunchKernelGGL((KernelConcatGrad<T>), dim3(grid_size), dim3(block_size), 0, context.stream(),
          input.data<T>(), input_row, input_col, outs_col_gpu,
          static_cast<int>(outputs_cols.size()), outs_gpu);
    }
  }
};

template class ConcatFunctor<platform::CUDADeviceContext, int>;
template class ConcatFunctor<platform::CUDADeviceContext, int64_t>;
template class ConcatFunctor<platform::CUDADeviceContext, float>;
template class ConcatFunctor<platform::CUDADeviceContext, double>;

template class ConcatGradFunctor<platform::CUDADeviceContext, int>;
template class ConcatGradFunctor<platform::CUDADeviceContext, int64_t>;
template class ConcatGradFunctor<platform::CUDADeviceContext, float>;
template class ConcatGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
