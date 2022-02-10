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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/bincount_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::PADDLE_CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T, typename InputT, typename OutT>
__global__ void KernelBincount(const InputT* input, const int total_elements,
                               const bool has_weights, const T* weights,
                               OutT* output) {
  if (!has_weights) {
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
      paddle::platform::CudaAtomicAdd(&output[input[i]], 1L);
    }
  } else {
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
      paddle::platform::CudaAtomicAdd(&output[input[i]],
                                      static_cast<OutT>(weights[i]));
    }
  }
}

template <typename DeviceContext, typename T, typename InputT>
void BincountCUDAInner(const framework::ExecutionContext& context) {
  const Tensor* input = context.Input<framework::Tensor>("X");
  const Tensor* weights = context.Input<framework::Tensor>("Weights");
  Tensor* output = context.Output<framework::Tensor>("Out");
  auto& minlength = context.Attr<int>("minlength");

  const InputT* input_data = input->data<InputT>();

  const int input_numel = input->numel();

  if (input_data == nullptr) {
    framework::DDim out_dim{0};
    output->Resize(out_dim);
    output->mutable_data<T>(context.GetPlace());
    return;
  }
  auto input_x = framework::EigenVector<InputT>::Flatten(*input);

  framework::Tensor input_min_t, input_max_t;
  auto* input_max_data =
      input_max_t.mutable_data<InputT>({1}, context.GetPlace());
  auto* input_min_data =
      input_min_t.mutable_data<InputT>({1}, context.GetPlace());

  auto input_max_scala = framework::EigenScalar<InputT>::From(input_max_t);
  auto input_min_scala = framework::EigenScalar<InputT>::From(input_min_t);

  auto* place = context.template device_context<DeviceContext>().eigen_device();
  input_max_scala.device(*place) = input_x.maximum();
  input_min_scala.device(*place) = input_x.minimum();

  Tensor input_min_cpu, input_max_cpu;
  paddle::framework::TensorCopySync(input_max_t, platform::CPUPlace(),
                                    &input_max_cpu);
  paddle::framework::TensorCopySync(input_min_t, platform::CPUPlace(),
                                    &input_min_cpu);

  InputT input_min = input_min_cpu.data<InputT>()[0];

  PADDLE_ENFORCE_GE(
      input_min, static_cast<InputT>(0),
      platform::errors::InvalidArgument(
          "The elements in input tensor must be non-negative ints"));

  int64_t output_size =
      static_cast<int64_t>(input_max_cpu.data<InputT>()[0]) + 1L;

  output_size = std::max(output_size, static_cast<int64_t>(minlength));
  framework::DDim out_dim{output_size};
  output->Resize(out_dim);

  bool has_weights = (weights != nullptr);

  const T* weights_data = has_weights ? weights->data<T>() : nullptr;

  auto stream =
      context.template device_context<platform::CUDADeviceContext>().stream();

  if (!has_weights) {
    int64_t* output_data = output->mutable_data<int64_t>(context.GetPlace());
    math::SetConstant<DeviceContext, int64_t>()(
        context.template device_context<DeviceContext>(), output, 0L);

    KernelBincount<T, InputT, int64_t><<<GET_BLOCKS(input_numel),
                                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        input_data, input_numel, has_weights, weights_data, output_data);
  } else {
    const auto& weights_type = weights->type();

    if (weights_type == framework::proto::VarType::FP32) {
      float* output_data = output->mutable_data<float>(context.GetPlace());
      math::SetConstant<DeviceContext, float>()(
          context.template device_context<DeviceContext>(), output,
          static_cast<float>(0));

      KernelBincount<T, InputT, float><<<GET_BLOCKS(input_numel),
                                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          input_data, input_numel, has_weights, weights_data, output_data);
    } else {
      double* output_data = output->mutable_data<double>(context.GetPlace());
      math::SetConstant<DeviceContext, double>()(
          context.template device_context<DeviceContext>(), output,
          static_cast<double>(0));

      KernelBincount<T, InputT, double><<<GET_BLOCKS(input_numel),
                                          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          input_data, input_numel, has_weights, weights_data, output_data);
    }
  }
}

template <typename DeviceContext, typename T>
class BincountCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<framework::Tensor>("X");
    const auto& input_type = input->type();

    if (input_type == framework::proto::VarType::INT32) {
      BincountCUDAInner<DeviceContext, T, int>(context);
    } else if (input_type == framework::proto::VarType::INT64) {
      BincountCUDAInner<DeviceContext, T, int64_t>(context);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    bincount, ops::BincountCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::BincountCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::BincountCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::BincountCUDAKernel<paddle::platform::CUDADeviceContext, double>);
