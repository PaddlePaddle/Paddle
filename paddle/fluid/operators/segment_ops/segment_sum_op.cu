/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/segment_ops/segment_sum_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"

namespace paddle {
namespace operators {

template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSegmentSumCustomKernel(const Index input_outer_dim_size,
                                             const Index inner_dim_size,
                                             const Index output_outer_dim_size,
                                             const Index* segment_ids,
                                             const T* input, T* output,
                                             const Index total_stripe_count) {
  CUDA_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id) {
          platform::CudaAtomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += input[(input_outer_dim_index_base + j) * inner_dim_size +
                   segment_offset];
      // sum += __ldg(input + (input_outer_dim_index_base + j) * inner_dim_size
      // +segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    platform::CudaAtomicAdd(output + output_index, sum);
  }
}

template <typename T, typename Index>
void SegmentSumCUDAFunctor(const platform::CUDADeviceContext& ctx,
                           const framework::Tensor& input,
                           const framework::Tensor& segment_ids,
                           framework::Tensor* output, const Index output_rows) {
  // Launch kernel to compute sorted segment sum.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = input.numel();
  const Index input_outer_dim_size = segment_ids.dims()[0];
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;

  const Index OuterDimTileSize = 8;

  const Index input_outer_dim_num_stripe =
      (input_outer_dim_size + OuterDimTileSize - 1) / OuterDimTileSize;

  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;

  auto config = platform::GetGpuLaunchConfig1D(ctx, total_stripe_count);

  SortedSegmentSumCustomKernel<T, Index, OuterDimTileSize><<<
      config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
      input_outer_dim_size, input_inner_dim_size, output_rows,
      segment_ids.data<Index>(), input.data<T>(), output->data<T>(),
      total_stripe_count);
}

template <typename DeviceContext, typename T, typename Index>
void SegmentSumCUDAKernelCompute(const framework::ExecutionContext& context) {
  auto* input = context.Input<framework::Tensor>("X");
  auto* segment_ids = context.Input<framework::Tensor>("SegmentIds");
  auto* output = context.Output<framework::Tensor>("Out");
  // todo: check wheather segment_ids is 1-d tensor
  // OP_REQUIRES_ASYNC(
  //    context, TensorShapeUtils::IsVector(segment_ids.shape()),
  //    errors::InvalidArgument("segment_ids should be a vector."), done);

  int64_t num_indices = segment_ids->numel();
  PADDLE_ENFORCE_EQ(
      num_indices, input->dims()[0],
      platform::errors::InvalidArgument(
          "segment_ids should be the same size as dimension 0 of input X."));

  // todo: check num_indices=0
  // if (num_indices == 0) {
  //  TensorShape output_shape = input.shape();
  //  output_shape.set_dim(0, 0);

  //  Tensor* output = nullptr;
  //  OP_REQUIRES_OK_ASYNC(
  //      context, context->allocate_output(0, output_shape, &output), done);
  //  done();
  //  return;
  //}

  // copy and get the length of final reduced tensor
  Tensor length;
  length.mutable_data<Index>(framework::make_ddim({1}), platform::CPUPlace());
  Index* length_data = length.data<Index>();
  const Index* segment_ids_data = segment_ids->data<Index>();

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(length_data, segment_ids_data + num_indices - 1, sizeof(Index),
                 cudaMemcpyDeviceToHost));

  Index length_host = length_data[0];
  length_host++;
  PADDLE_ENFORCE_GT(
      length_host, 0,
      platform::errors::InvalidArgument(
          "segment ids must be >= 0, but got last id %d", length_data[0]));
  auto dims = input->dims();
  dims[0] = static_cast<int64_t>(length_host);
  output->Resize({dims});
  output->mutable_data<T>(context.GetPlace());

  if (output->numel() == 0) {
    return;
  }
  // Set 'output' to zeros.
  math::SetConstant<DeviceContext, T> set_zero;
  auto& dev_ctx = context.template device_context<DeviceContext>();
  set_zero(dev_ctx, output, static_cast<T>(0));
  if (input->numel() == 0 || segment_ids->numel() == 0) {
    return;
  }

  SegmentSumCUDAFunctor<T, Index>(dev_ctx, *input, *segment_ids, output,
                                  length_host);
  // functor::SegmentSumFunctor<T, Index> functor_;
}

template <typename DeviceContext, typename T>
class SegmentSumCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* segment = context.Input<framework::Tensor>("SegmentIds");
    auto index_type = segment->type();
    if (index_type == framework::proto::VarType::INT32) {
      SegmentSumCUDAKernelCompute<DeviceContext, T, int>(context);
    } else if (index_type == framework::proto::VarType::INT64) {
      SegmentSumCUDAKernelCompute<DeviceContext, T, int64_t>(context);
    } else {
      PADDLE_THROW("unsupported index type");
    }
  }
};

template <typename DeviceContext, typename T>
class SegmentSumGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* segment = context.Input<Tensor>("SegmentIds");
    auto* out_g = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<Tensor>(framework::GradVarName("X"));

    in_g->mutable_data<T>(context.GetPlace());

    auto index_type = segment->type();
    if (index_type == framework::proto::VarType::INT32) {
      GPUGather<T, int>(context.device_context(), *out_g, *segment, in_g);
    } else if (index_type == framework::proto::VarType::INT64) {
      GPUGather<T, int64_t>(context.device_context(), *out_g, *segment, in_g);
    } else {
      PADDLE_THROW("unsupported index type");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(segment_sum, ops::SegmentSumCUDAKernel<CUDA, float>,
                        ops::SegmentSumCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(segment_sum_grad,
                        ops::SegmentSumGradCUDAKernel<CUDA, float>,
                        ops::SegmentSumGradCUDAKernel<CUDA, double>);
