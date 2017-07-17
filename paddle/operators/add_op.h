#pragma once
#include <glog/logging.h>
#include <paddle/framework/operator.h>

namespace paddle {
namespace operators {

template <typename Place>
class AddKernel : public framework::OpKernel {
public:
  void Compute(const KernelContext &context) const override {
    LOG(INFO) << "Add kernel in " << typeid(Place).name();
  }
};

}  // namespace op
}  // namespace paddle
