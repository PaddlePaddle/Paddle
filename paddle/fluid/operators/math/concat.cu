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

#include "paddle/fluid/operators/math/concat.h"
#include "paddle/fluid/platform/cuda_helper.h"

namespace paddle {
namespace operators {
namespace math {

// TODO(zcd): This can be replaced by tensor,
// if that, maybe we should add int8 to VarType::Type.
// Or replaced by tensorArray.
static constexpr int MaxSize = 8;
template <typename T>
struct CUDADeviceArray {
  T data[MaxSize];
  int size;
};

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
__global__ void KernelConcat(const CUDADeviceArray<const T*> inputs,
                             const CUDADeviceArray<int> input_cols,
                             const int output_rows, const int output_cols,
                             T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int segment = upper_bound<int>(input_cols.data, input_cols.size, tid_x) - 1;

  int curr_offset = input_cols.data[segment];
  int curr_segment = segment;
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    T curr_col_offset;
    while ((curr_col_offset = input_cols.data[curr_segment + 1]) <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    const T* input_ptr = inputs.data[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * segment_width + local_col];
  }
}

template <typename T>
__global__ void KernelConcat(const CUDADeviceArray<const T*> inputs,
                             const int input_col, const int output_rows,
                             const int output_cols, T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  float inv_input_col = 1.0 / input_col;
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * inv_input_col;
    int in_offset = tid_x - split * input_col;
    const T* input_ptr = inputs.data[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * input_col + in_offset];
  }
}

template <typename T>
__global__ void KernelConcatGrad(const T* input, const int input_row,
                                 const int input_col,
                                 CUDADeviceArray<int> output_cols,
                                 CUDADeviceArray<T*> outputs) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int segment = upper_bound<int>(output_cols.data, output_cols.size, tid_x) - 1;
  int curr_offset = output_cols.data[segment];
  int curr_segment = segment;
  for (; tid_x < input_col; tid_x += blockDim.x * gridDim.x) {
    T curr_col_offset;
    while ((curr_col_offset = output_cols.data[curr_segment + 1]) <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs.data[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < input_row; tid_y += blockDim.y * gridDim.y)
      output_ptr[tid_y * segment_width + local_col] =
          input[tid_y * input_col + tid_x];
  }
}

template <typename T>
__global__ void KernelConcatGrad(const T* input, const int input_row,
                                 const int input_col, const int output_cols,
                                 CUDADeviceArray<T*> outputs) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  float inv_input_col = 1.0 / input_col;
  for (; tid_x < input_col; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * inv_input_col;
    int in_offset = tid_x - split * input_col;
    T* output_ptr = outputs.data[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < input_row; tid_y += blockDim.y * gridDim.y)
      output_ptr[tid_y * output_cols + in_offset] =
          input[tid_y * input_col + tid_x];
  }
}

/*
 * All tensors' dimension should be the same.
 */
template <typename T>
class ConcatFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::vector<framework::Tensor>& input, const int axis,
                  framework::Tensor* output) {
    // assume the the max size of input is less than 8 and see the performance
    // save origin dim
    int num = input.size();
    PADDLE_ENFORCE_LT(num, MaxSize, "input number should be less than %d",
                      MaxSize);
    // get the matrix size
    int rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int cols = input[0].numel() / rows;
    int out_rows = rows, out_cols = 0;
    bool sameShape = true;

    CUDADeviceArray<const T*> inputs_data;
    CUDADeviceArray<int> inputs_cols;
    inputs_data.size = num;
    inputs_cols.size = num + 1;
    inputs_cols.data[0] = 0;
    // reshape to matrix
    // check input shape is valid
    for (int i = 0; i < num; ++i) {
      int t_cols = input[i].numel() / rows;
      if (sameShape) {
        if (t_cols != cols) sameShape = false;
      }
      out_cols += t_cols;
      inputs_cols.data[i + 1] = out_cols;
      inputs_data.data[i] = input[i].data<T>();
    }

    // computation
    // set the thread block and grid according to CurrentDeviceId
    const int kThreadsPerBlock = 1024;
    int block_cols = std::min(out_cols, kThreadsPerBlock);
    int block_rows = std::max(kThreadsPerBlock / block_cols, 1);
    dim3 block_size = dim3(block_cols, block_rows, 1);

    int dev_id = paddle::platform::GetCurrentDeviceId();
    int multi_process = paddle::platform::GetCUDAMultiProcessors(dev_id);
    int max_threads_per_mp =
        paddle::platform::GetCUDAMaxThreadsPerMultiProcessor(dev_id);
    int max_threads = multi_process * max_threads_per_mp;
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    int grid_cols =
        std::min((out_cols + block_cols - 1) / block_cols, max_blocks);
    int grid_rows =
        std::min(max_blocks / grid_cols, std::max(out_rows / block_rows, 1));
    dim3 grid_size = dim3(grid_cols, grid_rows, 1);

    if (sameShape) {
      KernelConcat<<<grid_size, block_size, 0, context.stream()>>>(
          inputs_data, cols, out_rows, out_cols, output->data<T>());
    } else {
      KernelConcat<<<grid_size, block_size, 0, context.stream()>>>(
          inputs_data, inputs_cols, out_rows, out_cols, output->data<T>());
    }
  }
};

template <typename T>
class ConcatGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const int axis,
                  std::vector<framework::Tensor>& outputs) {
    // assume the the max size of input is less than 8 and see the performance
    // save origin dim
    int num = outputs.size();
    PADDLE_ENFORCE_LT(num, MaxSize, "input number should be less than %d",
                      MaxSize);

    // get the matrix size
    int input_row = 1;
    auto dim_0 = outputs[0].dims();
    for (int i = 0; i < axis; ++i) {
      input_row *= dim_0[i];
    }

    int output_col_0 = outputs[0].numel() / input_row;
    int input_col = 0;
    bool sameShape = true;

    CUDADeviceArray<T*> outputs_data;
    CUDADeviceArray<int> outputs_cols;
    outputs_data.size = num;
    outputs_cols.size = num + 1;
    outputs_cols.data[0] = 0;

    for (int i = 0; i < num; ++i) {
      int t_col = outputs[i].numel() / input_row;
      if (sameShape) {
        if (t_col != output_col_0) sameShape = false;
      }
      input_col += t_col;
      outputs_cols.data[i + 1] = input_col;
      outputs_data.data[i] = outputs[i].data<T>();
    }

    // computation
    const int kThreadsPerBlock = 256;
    int block_cols = std::min(input_col, kThreadsPerBlock);
    int block_rows = std::max(kThreadsPerBlock / block_cols, 1);
    dim3 block_size = dim3(block_cols, block_rows, 1);

    int grid_cols = (input_col + block_cols - 1) / block_cols;
    int grid_rows = (input_row + block_rows - 1) / block_rows;
    dim3 grid_size = dim3(grid_cols, grid_rows, 1);

    if (sameShape) {
      KernelConcatGrad<<<grid_size, block_size, 0, context.stream()>>>(
          input.data<T>(), input_row, input_col, output_col_0, outputs_data);
    } else {
      KernelConcatGrad<<<grid_size, block_size, 0, context.stream()>>>(
          input.data<T>(), input_row, input_col, outputs_cols, outputs_data);
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
