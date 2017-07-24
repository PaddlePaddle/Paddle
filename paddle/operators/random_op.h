#pragma once
#include <random>
#include "glog/logging.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {
template <typename Place, typename T>
class RandomOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& context) const override {
    auto* output = context.Output(0)->GetMutable<framework::Tensor>();
    output->mutable_data<T>(context.GetPlace());

    auto shape = context.op_.attrs_.at("Shape");
    auto mean = context.op_.attrs_.at("mean");
    auto std = context.op_.attrs_.at("std");
    auto seed = context.op_.attrs_.at("seed");
    // std::default_random_engine generator(seed);
    // std::normal_distribution<double> distribution(mean, std);

    framework::EigenMatrix<T>::From(*output).device(*(
        context.GetEigenDevice<Place>())) = framework::EigenMatrix<T>::Random();
  }
};

}  // namespace operators
}  // namespace paddle
