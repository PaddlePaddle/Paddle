#pragma once
#include <glog/logging.h>
#include <paddle/framework/operator.h>

namespace paddle {
namespace operators {

template <typename Place>
class MomentumSGDOpKernel : public framework::OpKernel {
public:
  void Compute(const framework::KernelContext &context) const override {
    LOG(INFO) << "Add kernel in " << typeid(Place).name();
  }
};

}  // namespace operators
}  // namespace paddle
