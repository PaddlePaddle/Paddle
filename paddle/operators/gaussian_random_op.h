#pragma once
#include <random>
#include "glog/logging.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class GaussianRandomOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext& context) const override {}
};

}  // namespace operators
}  // namespace paddle
