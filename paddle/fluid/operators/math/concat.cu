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
__global__ void KernelConcat(T** inputs_data, const int fixed_in_col,
                             const int out_rows, const int out_cols,
                             T* output_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < out_cols; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * 1.0 / fixed_in_col;
    int in_offset = tid_x - split * fixed_in_col;
    T* input_ptr = inputs_data[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < out_rows; tid_y += blockDim.y * gridDim.y) {
      output_data[tid_y * out_cols + tid_x] =
          input_ptr[tid_y * fixed_in_col + in_offset];
    }
  }
}

template <typename T>
__global__ void KernelConcatGrad(const T* input_data, const int in_row,
                                 const int in_col, const int* out_cols,
                                 int out_cols_size, T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int segment = upper_bound<int>(out_cols, out_cols_size, tid_x) - 1;
  int curr_offset = out_cols[segment];
  int curr_segment = segment;
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    T curr_col_offset;
    while ((curr_col_offset = out_cols[curr_segment + 1]) <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs_data[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
      output_ptr[tid_y * segment_width + local_col] =
          input_data[tid_y * in_col + tid_x];
  }
}

template <typename T>
__global__ void KernelConcatGrad(const T* input_data, const int in_row,
                                 const int in_col, const int fixed_out_col,
                                 T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x / fixed_out_col;
    int in_offset = tid_x - split * fixed_out_col;
    T* output_ptr = outputs_data[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
      output_ptr[tid_y * fixed_out_col + in_offset] =
          input_data[tid_y * in_col + tid_x];
  }
}

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::vector<framework::Tensor>& input, const int axis,
                  framework::Tensor* output) {
    // TODO(zcd): Add input data validity checking
    int in_num = input.size();
    int in_row = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      in_row *= dim_0[i];
    }
    int in_col = input[0].numel() / in_row;
    int out_row = in_row, out_col = 0;

    framework::Vector<int16_t> inputs_data(in_num * sizeof(T*) / 2);
    framework::Vector<int> inputs_col(in_num + 1);
    T** inputs_ptr = reinterpret_cast<T**>(inputs_data.data());

    inputs_col[0] = 0;
    bool sameShape = true;
    for (int i = 0; i < in_num; ++i) {
      int t_cols = input[i].numel() / in_row;
      if (sameShape) {
        if (t_cols != in_col) sameShape = false;
      }
      out_col += t_cols;
      inputs_col[i + 1] = out_col;
      inputs_ptr[i] = const_cast<T*>(input[i].data<T>());
    }

    T** dev_ins_data =
        reinterpret_cast<T**>(inputs_data.CUDAMutableData(context.GetPlace()));

    // computation
    // set the thread block and grid according to CurrentDeviceId
    const int kThreadsPerBlock = 1024;
    int block_cols = kThreadsPerBlock;
    if (out_col < kThreadsPerBlock) {  // block_cols is aligned by 32.
      block_cols = ((out_col + 31) >> 5) << 5;
    }
    int block_rows = kThreadsPerBlock / block_cols;
    dim3 block_size = dim3(block_cols, block_rows, 1);

    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    int grid_cols =
        std::min((out_col + block_cols - 1) / block_cols, max_blocks);
    int grid_rows =
        std::min(max_blocks / grid_cols, std::max(out_row / block_rows, 1));
    dim3 grid_size = dim3(grid_cols, grid_rows, 1);

    if (sameShape) {
      KernelConcat<<<grid_size, block_size, 0, context.stream()>>>(
          dev_ins_data, in_col, out_row, out_col, output->data<T>());
    } else {
      const int* dev_ins_col_data = inputs_col.CUDAData(context.GetPlace());
      KernelConcat<<<grid_size, block_size, 0, context.stream()>>>(
          dev_ins_data, dev_ins_col_data, static_cast<int>(inputs_col.size()),
          out_row, out_col, output->data<T>());
    }
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatGradFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input, const int axis,
                  std::vector<framework::Tensor>& outputs) {
    // TODO(zcd): Add input data validity checking
    int o_num = outputs.size();
    int out_row = 1;
    auto dim_0 = outputs[0].dims();
    for (int i = 0; i < axis; ++i) {
      out_row *= dim_0[i];
    }

    int out_col = outputs[0].numel() / out_row;
    int in_col = 0, in_row = out_row;
    bool sameShape = true;

    framework::Vector<int16_t> outputs_data(o_num * sizeof(T*) / 2);
    framework::Vector<int> outputs_cols(o_num + 1);
    T** outputs_ptr = reinterpret_cast<T**>(outputs_data.data());

    outputs_cols[0] = 0;
    for (int i = 0; i < o_num; ++i) {
      int t_col = outputs[i].numel() / out_row;
      if (sameShape) {
        if (t_col != out_col) sameShape = false;
      }
      in_col += t_col;
      outputs_cols[i + 1] = in_col;
      outputs_ptr[i] = outputs[i].data<T>();
    }

    T** dev_out_gpu_data =
        reinterpret_cast<T**>(outputs_data.CUDAMutableData(context.GetPlace()));

    // computation
    const int kThreadsPerBlock = 1024;
    int block_cols = kThreadsPerBlock;
    if (in_col < kThreadsPerBlock) {  // block_cols is aligned by 32.
      block_cols = ((in_col + 31) >> 5) << 5;
    }
    int block_rows = kThreadsPerBlock / block_cols;
    dim3 block_size = dim3(block_cols, block_rows, 1);

    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    int grid_cols =
        std::min((in_col + block_cols - 1) / block_cols, max_blocks);
    int grid_rows =
        std::min(max_blocks / grid_cols, std::max(out_row / block_rows, 1));
    dim3 grid_size = dim3(grid_cols, grid_rows, 1);

    if (sameShape) {
      KernelConcatGrad<<<grid_size, block_size, 0, context.stream()>>>(
          input.data<T>(), in_row, in_col, out_col, dev_out_gpu_data);
    } else {
      const int* dev_outs_col_data = outputs_cols.CUDAData(context.GetPlace());
      KernelConcatGrad<<<grid_size, block_size, 0, context.stream()>>>(
          input.data<T>(), in_row, in_col, dev_outs_col_data,
          static_cast<int>(outputs_cols.size()), dev_out_gpu_data);
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
