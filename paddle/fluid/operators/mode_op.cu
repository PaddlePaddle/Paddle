// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mode_op.h"
#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/fluid/operators/top_k_v2_op.h"

namespace paddle {
namespace operators {

int ComputeBlockSize(int col) {
  if (col > 512)
    return 1024;
  else if (col > 256 && col <= 512)
    return 512;
  else if (col > 128 && col <= 256)
    return 256;
  else if (col > 64 && col <= 128)
    return 128;
  else
    return 64;
}

template <typename T>
void getModebySort(const platform::CUDADeviceContext& ctx,
                   const framework::Tensor* input_tensor,
                   const int64_t num_cols, const int64_t num_rows,
                   T* out_tensor, int64_t* indices_tensor) {
  framework::Tensor input_tmp;
  framework::TensorCopy(*input_tensor, ctx.GetPlace(), &input_tmp);
  T* input_tmp_data = input_tmp.mutable_data<T>(ctx.GetPlace());
  input_tmp.Resize(framework::make_ddim({num_rows, num_cols}));
  thrust::device_ptr<T> out_tensor_ptr(out_tensor);
  thrust::device_ptr<int64_t> indices_tensor_ptr(indices_tensor);

  for (int64_t i = 0; i < num_rows; ++i) {
    T* begin = input_tmp_data + num_cols * i;
    T* end = input_tmp_data + num_cols * (i + 1);
    thrust::device_vector<int64_t> indices_data(num_cols);
    thrust::sequence(thrust::device, indices_data.begin(),
                     indices_data.begin() + num_cols);
    thrust::sort_by_key(thrust::device, begin, end, indices_data.begin());
    int unique = 1 + thrust::inner_product(thrust::device, begin, end - 1,
                                           begin + 1, 0, thrust::plus<int>(),
                                           thrust::not_equal_to<T>());
    thrust::device_vector<T> keys_data(unique);
    thrust::device_vector<int64_t> cnts_data(unique);
    thrust::reduce_by_key(thrust::device, begin, end,
                          thrust::constant_iterator<int>(1), keys_data.begin(),
                          cnts_data.begin());
    auto it = thrust::max_element(thrust::device, cnts_data.begin(),
                                  cnts_data.begin() + unique);
    T mode = keys_data[it - cnts_data.begin()];
    int64_t counts = cnts_data[it - cnts_data.begin()];
    auto pos = thrust::find(thrust::device, begin, end, mode);
    int64_t index = indices_data[pos - begin + counts - 1];
    out_tensor_ptr[i] = static_cast<T>(mode);
    indices_tensor_ptr[i] = static_cast<int64_t>(index);
  }
}

template <typename DeviceContext, typename T>
class ModeOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    bool keepdim = static_cast<bool>(ctx.Attr<bool>("keepdim"));

    // get the input dims
    const auto& in_dims = input->dims();
    // calcluate the real axis
    if (axis < 0) axis += in_dims.size();

    auto out_dims = output->dims();

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    if (axis == in_dims.size() - 1) {
      const int64_t& input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t& input_width = in_dims[in_dims.size() - 1];
      const auto& dev_ctx = ctx.cuda_device_context();
      getModebySort<T>(dev_ctx, input, input_width, input_height, output_data,
                       indices_data);
    } else {
      std::vector<int> trans_axis;
      for (int i = 0; i < axis; i++) {
        trans_axis.emplace_back(i);
      }
      trans_axis.emplace_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans_axis.emplace_back(i);
      }
      trans_axis.emplace_back(axis);

      if (!keepdim) {
        std::vector<int> tmp_out_shape;
        for (int i = 0; i < axis; i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        tmp_out_shape.emplace_back(1);
        for (int i = axis + 1; i < in_dims.size(); i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        framework::DDim tmp_out_dim = framework::make_ddim(tmp_out_shape);
        output->Resize(tmp_out_dim);
        indices->Resize(tmp_out_dim);
      }

      framework::DDim trans_shape(in_dims);
      framework::DDim trans_out_shape(in_dims);
      for (int i = 0; i < trans_axis.size(); i++) {
        trans_shape[i] = in_dims[trans_axis[i]];
        trans_out_shape[i] = in_dims[trans_axis[i]];
      }
      trans_out_shape[in_dims.size() - 1] = 1;

      // second step, tranpose the input
      framework::Tensor trans_input;
      trans_input.mutable_data<T>(trans_shape, ctx.GetPlace());
      int ndims = trans_axis.size();
      const auto& dev_ctx = ctx.cuda_device_context();
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, *input,
                                                   &trans_input, trans_axis);
      framework::Tensor trans_ind;
      int64_t* trans_ind_data =
          trans_ind.mutable_data<int64_t>(trans_out_shape, ctx.GetPlace());
      framework::Tensor trans_out;
      T* trans_out_data =
          trans_out.mutable_data<T>(trans_out_shape, ctx.GetPlace());

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_shape, 0, trans_shape.size() - 1));
      const int64_t input_width = trans_shape[trans_shape.size() - 1];
      getModebySort<T>(dev_ctx, &trans_input, input_width, input_height,
                       trans_out_data, trans_ind_data);
      // last step, tranpose back the indices and output
      TransCompute<platform::CUDADeviceContext, int64_t>(
          ndims, dev_ctx, trans_ind, indices, trans_axis);
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, trans_out,
                                                   output, trans_axis);
      if (!keepdim) {
        output->Resize(out_dims);
        indices->Resize(out_dims);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ModeOpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* x = context.Input<framework::Tensor>("X");
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<framework::Tensor>("Indices");
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    int axis = context.Attr<int>("axis");

    const auto& in_dims = x->dims();
    auto out_dims = indices->dims();

    if (axis < 0) axis += in_dims.size();
    // allocate the cuda memory for the x_grad
    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();

    int pre, n, post;
    GetDims(in_dims, axis, &pre, &n, &post);

    // calcluate the block and grid num
    auto& dev_ctx = context.cuda_device_context();
    int block_size = ComputeBlockSize(post);
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
    int grid_size = std::min(max_blocks, pre);
    AssignGradWithAxis<T><<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
        out_grad_data, indices_data, x_grad_data, pre, post, n, 1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    mode, ops::ModeOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ModeOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ModeOpCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ModeOpCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mode_grad,
    ops::ModeOpGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ModeOpGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ModeOpGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ModeOpGradCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
