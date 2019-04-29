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

#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void ConcatKernel(T** inputs, const int64_t* input_cols,
                             int col_size, const int output_rows,
                             const int output_cols, T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_segment = 0;
  int curr_offset = input_cols[0];
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset = input_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = input_cols[curr_segment + 1];
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
__global__ void ConcatKernel(T** inputs_data, const int fixed_in_col,
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
__global__ void SplitKernel(const T* input_data, const int in_row,
                            const int in_col, const int64_t* out_cols,
                            int out_cols_size, T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_segment = 0;
  int curr_offset = out_cols[0];
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset = out_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = out_cols[curr_segment + 1];
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;
    T* output_ptr = outputs_data[curr_segment];
    if (output_ptr != nullptr) {
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * segment_width + local_col] =
            input_data[tid_y * in_col + tid_x];
    }
  }
}

template <typename T>
__global__ void SplitKernel(const T* input_data, const int in_row,
                            const int in_col, const int fixed_out_col,
                            T** outputs_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < in_col; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x / fixed_out_col;
    int in_offset = tid_x - split * fixed_out_col;
    T* output_ptr = outputs_data[split];
    if (output_ptr != nullptr) {
      int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
      for (; tid_y < in_row; tid_y += blockDim.y * gridDim.y)
        output_ptr[tid_y * fixed_out_col + in_offset] =
            input_data[tid_y * in_col + tid_x];
    }
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
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output, framework::Tensor* ins_info) {
    // TODO(zcd): Add input data validity checking
    int in_num = input.size();
    int in_row = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      in_row *= dim_0[i];
    }
    int in_col = input[0].numel() / in_row;
    int out_row = in_row, out_col = 0;

    std::vector<int64_t> ins_info_vec(2 * in_num + 1);

    ins_info_vec[in_num] = 0;
    bool sameShape = true;
    for (int i = 0; i < in_num; ++i) {
      int t_cols = input[i].numel() / in_row;
      if (sameShape) {
        if (t_cols != in_col) sameShape = false;
      }
      out_col += t_cols;
      ins_info_vec[in_num + i + 1] = out_col;
      ins_info_vec[i] = reinterpret_cast<int64_t>(input[i].data<T>());
    }

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

    int64_t ins_info_data_length = sameShape ? in_num : (2 * in_num + 1);
    // When ins_info is null, we use a temporary allocation.
    memory::AllocationPtr tmp_dev_ins_data;
    int64_t* dev_ins_info_data;
    if (ins_info) {
      dev_ins_info_data = ins_info->mutable_data<int64_t>(
          framework::make_ddim({ins_info_data_length}), context.GetPlace());
    } else {
      tmp_dev_ins_data =
          platform::DeviceTemporaryAllocator::Instance().Get(context).Allocate(
              ins_info_data_length * sizeof(int64_t));
      dev_ins_info_data = static_cast<int64_t*>(tmp_dev_ins_data->ptr());
    }
    memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                 dev_ins_info_data, platform::CPUPlace(),
                 static_cast<void*>(ins_info_vec.data()),
                 ins_info_data_length * sizeof(int64_t), context.stream());

    T** dev_ins_data = reinterpret_cast<T**>(dev_ins_info_data);

    if (sameShape) {
      ConcatKernel<<<grid_size, block_size, 0, context.stream()>>>(
          dev_ins_data, in_col, out_row, out_col, output->data<T>());
    } else {
      int64_t* dev_ins_col_data = dev_ins_info_data + in_num;
      ConcatKernel<<<grid_size, block_size, 0, context.stream()>>>(
          dev_ins_data, dev_ins_col_data, in_num + 1, out_row, out_col,
          output->data<T>());
    }
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  int axis, std::vector<framework::Tensor*>* outputs,
                  framework::Tensor* outs_info) {
    // TODO(zcd): Add input data validity checking
    int o_num = outputs->size();
    int out_row = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      out_row *= dim_0[i];
    }

    int out0_col = ref_inputs[0]->numel() / out_row;
    int in_col = 0, in_row = out_row;
    bool sameShape = true;

    std::vector<int64_t> outs_info_vec(2 * o_num + 1);

    outs_info_vec[o_num] = 0;
    for (int i = 0; i < o_num; ++i) {
      int t_col = ref_inputs.at(i)->numel() / out_row;
      if (sameShape) {
        if (t_col != out0_col) sameShape = false;
      }
      in_col += t_col;
      outs_info_vec[o_num + i + 1] = in_col;
      if (outputs->at(i) != nullptr) {
        outs_info_vec[i] = reinterpret_cast<int64_t>(outputs->at(i)->data<T>());
      } else {
        outs_info_vec[i] = 0;
      }
    }

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

    int64_t outs_info_data_length = sameShape ? o_num : (2 * o_num + 1);
    // When outs_info is null, we use a temporary allocation.
    memory::AllocationPtr tmp_dev_outs_data;
    int64_t* dev_outs_info_data;
    if (outs_info) {
      dev_outs_info_data = outs_info->mutable_data<int64_t>(
          framework::make_ddim({outs_info_data_length}), context.GetPlace());
    } else {
      tmp_dev_outs_data =
          platform::DeviceTemporaryAllocator::Instance().Get(context).Allocate(
              outs_info_data_length * sizeof(int64_t));
      dev_outs_info_data = static_cast<int64_t*>(tmp_dev_outs_data->ptr());
    }
    memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                 dev_outs_info_data, platform::CPUPlace(),
                 static_cast<void*>(outs_info_vec.data()),
                 outs_info_data_length * sizeof(int64_t), context.stream());

    T** dev_out_gpu_data = reinterpret_cast<T**>(dev_outs_info_data);

    if (sameShape) {
      SplitKernel<<<grid_size, block_size, 0, context.stream()>>>(
          input.data<T>(), in_row, in_col, out0_col, dev_out_gpu_data);
    } else {
      int64_t* dev_outs_col_data = dev_outs_info_data + o_num;
      SplitKernel<<<grid_size, block_size, 0, context.stream()>>>(
          input.data<T>(), in_row, in_col, dev_outs_col_data, o_num + 1,
          dev_out_gpu_data);
    }
  }
};

#define DEFINE_FUNCTOR(type)                                       \
  template class ConcatFunctor<platform::CUDADeviceContext, type>; \
  template class SplitFunctor<platform::CUDADeviceContext, type>

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace operators
}  // namespace paddle
