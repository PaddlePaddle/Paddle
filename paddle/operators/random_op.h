#pragma once
#include <random>
#include "glog/logging.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

template <typename T>
bool Gaussian(platform::CPUDeviceContext* ctx,
              T* output,
              const int size,
              const T& mean,
              const T& std,
              const T& seed) {
  auto g = ctx->RandGenerator(seed);
  std::normal_distribution<T> distribution(mean, std);
  for (int i = 0; i < size; ++i) {
    output[i] = distribution(g);
  }
  return true;
}

#ifndef PADDLE_ONLY_CPU
template <typename T>
bool Gaussian(platform::CUDADeviceContext* ctx,
              T* output,
              const int size,
              const T& mean,
              const T& std,
              const T& seed) {
  auto g = ctx->RandGenerator(seed);
  return curandGenerateNormal(g, output, size, mean, std);
}
#endif

template <typename Place, typename T>
class RandomOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& context) const override {
    auto mean = context.op_.GetAttr<T>("mean");
    auto std = context.op_.GetAttr<T>("std");
    auto seed = context.op_.GetAttr<T>("seed");
    auto* output = context.Output(0)->GetMutable<framework::Tensor>();
    auto place = context.GetPlace();
    if (platform::is_cpu_place(place)) {
      Gaussian(
          dynamic_cast<platform::CPUDeviceContext*>(context.device_context_),
          output->mutable_data<T>(context.GetPlace()),
          framework::product(output->dims()),
          mean,
          std,
          seed);
    } else {
#ifndef PADDLE_ONLY_CPU
      Gaussian(
          dynamic_cast<platform::CUDADeviceContext*>(context.device_context_),
          output->mutable_data<T>(context.GetPlace()),
          framework::product(output->dims()),
          mean,
          std,
          seed);
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle
