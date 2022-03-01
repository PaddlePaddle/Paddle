
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

namespace phi {

template <typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& out_grad,
                       const std::vector<int64_t>& dims,
                       bool keep_dim,
                       bool reduce_all,
                       DenseTensor* x_grad) {

    if (context.GetPlace().GetType() == platform::CPUPlace().GetType() &&
        dims.size() == 1) {
      int in_dtype = context.Attr<int>("out_dtype");

      if (in_dtype >= 0) {
        Tensor tmp_tensor;
        auto* pre_input = context.Input<Tensor>(framework::GradVarName("Out"));
        auto in_kernel_type = framework::OpKernelType(
            framework::TransToProtoVarType(pre_input->dtype()),
            context.GetPlace());
        auto out_kernel_type = framework::OpKernelType(
            static_cast<framework::proto::VarType::Type>(in_dtype),
            context.GetPlace());
        framework::TransDataType(in_kernel_type, out_kernel_type, *pre_input,
                                 &tmp_tensor);
        ComputeFromInput(&tmp_tensor, context);
      } else {
        auto* input2 = context.Input<Tensor>(framework::GradVarName("Out"));
        ComputeFromInput(input2, context);
      }
      return;
    }
    // default use Eigen broadcast
    ReduceGradKernel<DeviceContext, T, Functor, kNoNeedBufferX> kernel;
    kernel.Compute(context);
}

} // namespace phi
