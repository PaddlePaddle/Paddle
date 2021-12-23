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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/kthvalue_function_cuda.h"
#include "paddle/fluid/operators/kthvalue_op.h"

namespace paddle {
namespace operators {

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

template <typename DeviceContext, typename T>
class KthvalueOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");

    // get the attributes
    int k = static_cast<int>(ctx.Attr<int>("k"));
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

      // The conclusion is drawn from the data through multiple sets of
      // statistics
      if (input_width >= 128 && k >= input_width * 0.75) {
        // if(true){
        if (SortKthvalue<T>(dev_ctx, input, input_width, input_height, k,
                            output, indices)) {
          // Successed, return.
          return;
        } else {
          LOG(INFO)
              << "KthvalueOP: Some errors happened when use cub sorting, use "
                 "default kthvalue kernel.";
        }
      }

      // NOTE: pass lds and dim same to input width.
      // NOTE: old matrix implementation of stride is different to eigen.
      const int kMaxHeight = 2048;
      int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
      switch (GetDesiredBlockDim(input_width)) {
#ifdef PADDLE_WITH_HIP
        FIXED_BLOCK_DIM(
            KeMatrixKthvalue<
                T, 20, kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                output_data, indices_data, input_data, input_width, input_width,
                static_cast<int>(k), gridx, input_height));
#else
        FIXED_BLOCK_DIM(
            KeMatrixKthvalue<
                T, 5, kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                output_data, indices_data, input_data, input_width, input_width,
                static_cast<int>(k), gridx, input_height));
#endif
        default:
          PADDLE_THROW(platform::errors::Fatal(
              "the input data shape has error in the kthvalue cuda kernel."));
      }
    } else {
      // first step, prepare the trans args for the tranpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);

      if (!keepdim) {
        std::vector<int> tmp_out_shape;
        for (int i = 0; i < axis; i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        tmp_out_shape.emplace_back(1);
        for (int i = axis + 1; i < in_dims.size(); i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        framework::DDim tmp_out_dims = framework::make_ddim(tmp_out_shape);
        output->Resize(tmp_out_dims);
        indices->Resize(tmp_out_dims);
      }

      framework::DDim trans_dims(in_dims);
      framework::DDim trans_out_dims(in_dims);
      for (int i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
        trans_out_dims[i] = in_dims[trans[i]];
      }
      trans_out_dims[in_dims.size() - 1] = 1;

      // second step, tranpose the input
      Tensor trans_input;
      trans_input.mutable_data<T>(trans_dims, ctx.GetPlace());
      int ndims = trans.size();
      const auto& dev_ctx = ctx.cuda_device_context();
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, *input,
                                                   &trans_input, trans);
      Tensor trans_ind;
      trans_ind.mutable_data<int64_t>(trans_out_dims, ctx.GetPlace());
      Tensor trans_out;
      trans_out.mutable_data<T>(trans_out_dims, ctx.GetPlace());

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];

      // The conclusion is drawn from the data through multiple sets of
      // statistics
      if (input_width >= 128 && k >= input_width * 0.75) {
        // if(true){
        if (SortKthvalue<T>(dev_ctx, &trans_input, input_width, input_height, k,
                            &trans_out, &trans_ind)) {
          // last step, tranpose back the indices and output
          TransCompute<platform::CUDADeviceContext, int64_t>(
              ndims, dev_ctx, trans_ind, indices, trans);
          TransCompute<platform::CUDADeviceContext, T>(
              ndims, dev_ctx, trans_out, output, trans);
          if (!keepdim) {
            output->Resize(out_dims);
            indices->Resize(out_dims);
          }
          return;
        } else {
          LOG(INFO)
              << "KthvalueOP: Some errors happened when use cub sorting, use "
                 "default kthvalue kernel.";
        }
      }

      const int kMaxHeight = 2048;
      int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
      switch (GetDesiredBlockDim(input_width)) {
#ifdef PADDLE_WITH_HIP
        FIXED_BLOCK_DIM(
            KeMatrixKthvalue<
                T, 20, kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                trans_out.data<T>(), trans_ind.data<int64_t>(),
                trans_input.data<T>(), input_width, input_width,
                static_cast<int>(k), gridx, input_height));
#else
        FIXED_BLOCK_DIM(
            KeMatrixKthvalue<
                T, 5, kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                trans_out.data<T>(), trans_ind.data<int64_t>(),
                trans_input.data<T>(), input_width, input_width,
                static_cast<int>(k), gridx, input_height));
#endif
        default:
          PADDLE_THROW(platform::errors::Fatal(
              "the input data shape has error in the kthvalue cuda kernel."));
      }

      // last step, tranpose back the indices and output
      TransCompute<platform::CUDADeviceContext, int64_t>(
          ndims, dev_ctx, trans_ind, indices, trans);
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, trans_out,
                                                   output, trans);
      if (!keepdim) {
        output->Resize(out_dims);
        indices->Resize(out_dims);
      }
    }
  }
};

#undef FIXED_BLOCK_DIM_BASE
#undef FIXED_BLOCK_DIM
template <typename DeviceContext, typename T>
class KthvalueOpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* x = context.Input<Tensor>("X");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    int axis = context.Attr<int>("axis");
    int k = static_cast<int>(context.Attr<int>("k"));

    const auto& in_dims = x->dims();
    auto out_dims = indices->dims();

    // get the real the axis and the k
    if (axis < 0) axis += in_dims.size();
    // allocate the cuda memory for the x_grad
    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();

    int pre, n, post;
    GetDims(in_dims, axis, &pre, &n, &post);

    // calcluate the block and grid num
    auto& dev_ctx = context.cuda_device_context();
    auto ComputeBlockSize = [](int col) {
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
    };
    int block_size = ComputeBlockSize(post * k);
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
    int grid_size = std::min(max_blocks, pre);

    // lanuch the cuda kernel to assign the grad
    AssignGradWithAxis<T><<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
        out_grad_data, indices_data, x_grad_data, pre, post, n, 1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    kthvalue,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    kthvalue_grad,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
