#pragma once
#include <random>
#include "glog/logging.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {
template <typename Place, typename T, typename Generator>
bool Gaussian(
    Generator g, T* output, const int size, const T& mean, const T& std);

template <typename Place, typename T>
class RandomOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& context) const override {
    auto mean = context.op_.attrs_.at("mean");
    auto std = context.op_.attrs_.at("std");
    auto seed = context.op_.attrs_.at("seed");
    auto* output = context.Output(0)->GetMutable<framework::Tensor>();
    output->mutable_data<T>(context.GetPlace());

    Gaussian<Place, T, >(, output, output->size(), mean, std) :
    // std::default_random_engine generator(seed);
    // std::normal_distribution<double> distribution(mean, std);

    // framework::EigenMatrix<T>::From(*output).device(*(
    //     context.GetEigenDevice<Place>())) =
    //     framework::EigenMatrix<T>::Random();
  }
};

// using paddle::platform::CPUPlace;
// template<CPUPlace, typename T>
// class RandomOpKernel : public framework::OpKernel {
// public:
//   void Compute(const framework::KernelContext& context) const override {

//     std::unique_ptr<default_random_engine> generator(seed);
//     for(size_t i=0; i < output->size(); ++i) {
//       output[i] = distribution(generator());
//     }
//   }

// };

// using paddle::platform::GPUPlace;
// template<GPUPlace, typename T>
// class RandomOpKernel : public framework::OpKernel {
// public:
//   void Compute(const framework::KernelContext& context) const override {

//   }
// }

}  // namespace operators
}  // namespace paddle
