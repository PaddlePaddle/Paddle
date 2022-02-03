
#include "paddle/pten/kernels/funcs/eigen/common.h"
#include "paddle/pten/kernels/batch_norm_kernel.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#include "paddle/fluid/operators/norm_utils.h"
#include "paddle/fluid/operators/norm_utils.cu.h"

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/pten/kernels/gpu/batch_norm_utils.h"
#include "paddle/fluid/platform/flags.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

namespace pten{


template <typename T, typename Context>
void BatchNormGradGradKernel(const Context& ctx,  
                    const DenseTensor& x_grad_grad, const DenseTensor& scale_grad_grad,
                    const DenseTensor& bias_grad_grad, const DenseTensor& y_grad,
                    const DenseTensor& x, \
                    const DenseTensor& scale, const DenseTensor& bias,
                    const DenseTensor& saved_mean, const DenseTensor& saved_variance,
                    paddle::optional<const DenseTensor&> mean,
                    paddle::optional<const DenseTensor&> variance,                    
                    float momentum, float epsilon, const std::string& data_layout_str,
                    bool is_test, bool use_global_stats, bool trainable_statistics,
                    bool fuse_with_relu,
                    DenseTensor* x_grad, DenseTensor* scale_grad, DenseTensor* y_grad_grad ){

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    const DataLayout data_layout =
        paddle::framework::StringToDataLayout(data_layout_str);
    
    const DenseTensor* running_mean = nullptr;
    const DenseTensor* running_variance = nullptr;
    if( use_global_stats )
    {
        running_mean = mean.get_ptr();
        running_variance = variance.get_ptr();
    }
    paddle::operators::NormDoubleGradFunctor<Context, T>(
        ctx, data_layout, &x, &scale, &y_grad, &saved_mean, &saved_variance, 
        running_mean, running_variance, epsilon,
        use_global_stats, &x_grad_grad, &scale_grad_grad, &bias_grad_grad, x_grad, scale_grad, y_grad_grad);


}
} //namespace pten

PT_REGISTER_KERNEL(batch_norm_grad_grad, GPU, ALL_LAYOUT, pten::BatchNormGradGradKernel, float, double) {}
