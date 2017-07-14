#pragma once
#include "glog/logging.h"
#include "paddle/framework/operator.h"
//#include "paddle/operators/add_op_functor.h"

namespace paddle {
namespace operators {

// Place can be CPUPlace or GPUPlace
template <typename Place, typename DataType>
class AddKernel : public framework::OpKernel {
public:
  void Compute(const KernelContext& context) const override {
    auto* input0 = context.Input(0);
    auto* input1 = context.Input(1);

    auto* output = context.Output(0);
    output->mutable_data<DataType>(Place());

    output->flat<T>().device(*(context.get_eigen_device<Place>())) =
        input0->flat<T>() + input1->flat<T>();
  }
};

}  // namespace operators
}  // namespace paddle
