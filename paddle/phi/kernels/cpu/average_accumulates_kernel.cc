#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/average_accumulates_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi{

template <>
void GetAccumulators<CPUContext>(const CPUContext& dev_ctx,
                                      const DenseTensor& in_num_accumulates,
                                      const DenseTensor& in_old_num_accumulates,
                                      const DenseTensor& in_num_updates,
                                      int64_t* num_updates,
                                      int64_t* num_accumulates,
                                      int64_t* old_num_accumulates) {
  *old_num_accumulates = in_old_num_accumulates.data<int64_t>()[0];
  *num_accumulates = in_num_accumulates.data<int64_t>()[0];
  *num_updates = in_num_updates.data<int64_t>()[0];
}

template <>
void SetAccumulators<CPUContext>(const CPUContext& dev_ctx,
                                      int64_t num_updates,
                                      int64_t num_accumulates,
                                      int64_t old_num_accumulates,
                                      DenseTensor* out_num_accumulates,
                                      DenseTensor* out_old_num_accumulates,
                                      DenseTensor* out_num_updates) {
  out_old_num_accumulates->data<int64_t>()[0] = old_num_accumulates;
  out_num_accumulates->data<int64_t>()[0] = num_accumulates;
  out_num_updates->data<int64_t>()[0] = num_updates;
}

template<typename T, typename Context>
void AverageAccumulatesKernel(const Context& dev_ctx,
                                const DenseTensor& param,
                                const DenseTensor& in_sum_1,
                                const DenseTensor& in_sum_2,
                                const DenseTensor& in_sum_3,
                                const DenseTensor& in_num_accumulates,
                                const DenseTensor& in_old_num_accumulates,
                                const DenseTensor& in_num_updates,
                                float average_window,
                                int64_t max_average_window,
                                int64_t min_average_window,
                                DenseTensor* out_sum_1,
                                DenseTensor* out_sum_2,
                                DenseTensor* out_sum_3,
                                DenseTensor* out_num_accumulates,
                                DenseTensor* out_old_num_accumulates,
                                DenseTensor* out_num_updates){
    // It is used to avoid loss of precision
    static const int64_t kMaxNumAccumulates = 16384;
    // Get accumulators from input
    int64_t num_updates = 0;
    int64_t num_accumulates = 0;
    int64_t old_num_accumulates = 0;
    GetAccumulators<Context>(
        dev_ctx, in_num_accumulates, in_old_num_accumulates, in_num_updates, &num_updates, &num_accumulates, &old_num_accumulates);

    // Get attrs
    // float average_window = ctx.Attr<float>("average_window");
    // int64_t max_average_window = ctx.Attr<int64_t>("max_average_window");
    // int64_t min_average_window = ctx.Attr<int64_t>("min_average_window");
    PADDLE_ENFORCE_LE(
        min_average_window,
        max_average_window,
        errors::InvalidArgument(
            "The min_average_window > "
            "max_average_window is not right, min_average_window is %ld, "
            "max_average_window is %ld.",
            min_average_window,
            max_average_window));

    // Get inputs
    //auto* param = ctx.Input<Tensor>("param");
    //auto* in_sum_1 = ctx.Input<Tensor>("in_sum_1");
    //auto* in_sum_2 = ctx.Input<Tensor>("in_sum_2");
    //auto* in_sum_3 = ctx.Input<Tensor>("in_sum_3");
    auto param_tensor = EigenVector<T>::Flatten(param);
    auto in_sum_1_tensor = EigenVector<T>::Flatten(in_sum_1);
    auto in_sum_2_tensor = EigenVector<T>::Flatten(in_sum_2);
    auto in_sum_3_tensor = EigenVector<T>::Flatten(in_sum_3);

    // Get outputs
    //auto* out_sum_1 = ctx.Output<Tensor>("out_sum_1");
    //auto* out_sum_2 = ctx.Output<Tensor>("out_sum_2");
    //auto* out_sum_3 = ctx.Output<Tensor>("out_sum_3");
    auto out_sum_1_tensor = EigenVector<T>::Flatten(*out_sum_1);
    auto out_sum_2_tensor = EigenVector<T>::Flatten(*out_sum_2);
    auto out_sum_3_tensor = EigenVector<T>::Flatten(*out_sum_3);

    // Compute
    //auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    auto& place = *dev_ctx.eigen_device();

    funcs::SetConstant<Context, T> constant_functor;
    ++num_updates;
    ++num_accumulates;
    out_sum_1_tensor.device(place) = in_sum_1_tensor + param_tensor;
    out_sum_2_tensor.device(place) = in_sum_2_tensor;
    out_sum_3_tensor.device(place) = in_sum_3_tensor;
    if (num_updates % kMaxNumAccumulates == 0) {
        // Move the sum to a different buffer to avoid loss of precision due to
        // too many sums.
        out_sum_2_tensor.device(place) = in_sum_2_tensor + in_sum_1_tensor;
        constant_functor(
            dev_ctx, out_sum_1, 0.0);
    }
    if (num_accumulates >= min_average_window &&
        num_accumulates >= std::min<int64_t>(max_average_window,
                                                num_updates * average_window)) {
        //  Now the average window is too long, discard the old sum.
        out_sum_3_tensor.device(place) = in_sum_1_tensor + in_sum_2_tensor;
        constant_functor(
            dev_ctx, out_sum_1, 0.0);
        constant_functor(
            dev_ctx, out_sum_2, 0.0);
        old_num_accumulates = num_accumulates;
        num_accumulates = 0;
    }

    // Set accumulators to output
    SetAccumulators<Context>(
        dev_ctx, num_updates, num_accumulates, old_num_accumulates, out_num_accumulates, out_old_num_accumulates, out_num_updates);        
}

} // namespace phi

PD_REGISTER_KERNEL(
    average_accumulates,
    CPU,
    ALL_LAYOUT,
    phi::AverageAccumulatesKernel,
    float,
    double){}