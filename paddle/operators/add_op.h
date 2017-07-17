#pragma once
#include "glog/logging.h"
#include "paddle/framework/operator.h"
//#include "paddle/operators/add_op_functor.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class AddKernel : public framework::OpKernel {
public:
  void Compute(const KernelContext& context) const override {
    auto input0 = context.Input(0)->Get<framework::Tensor>();
    auto input1 = context.Input(1)->Get<framework::Tensor>();
    auto* output = context.Output(0)->GetMutable<framework::Tensor>();

    output->mutable_data<T>(Place());

    output->flat<T>().device(*(context.get_eigen_device<Place>())) =
        input0.flat<T>() + input1.flat<T>();
  }
};

}  // namespace operators
}  // namespace paddle
