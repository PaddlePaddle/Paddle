#include "paddle/phi/kernels/lsqplus_grad_kernel.h"

namespace phi {
template <typename T, typename Context>
void LsqplusGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& alpha,
                       const DenseTensor& beta,
                       const DenseTensor& g,
                       // const DenseTensor& out,
                       const DenseTensor& out_grad,
                       int Qn,
                       int Qp,
                       DenseTensor* x_grad,
                       DenseTensor* alpha_grad,
                       DenseTensor* beta_grad) {
  // 分配空间
  dev_ctx.template Alloc<T>(alpha_grad);
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(beta_grad);

  // 计算中间变量
  DenseTensor alpha_matrix = FullLike<T, Context>(dev_ctx, x, 0);
  DenseTensor beta_matrix = FullLike<T, Context>(dev_ctx, x, 0);
  DenseTensor mask = FullLike<T, Context>(dev_ctx, x, 0);
  // gradient scaling
  DenseTensor gradient_scale = FullLike<T, Context>(dev_ctx, x, 0);
  MultiplyKernel<T, Context>(dev_ctx, out_grad, g, &gradient_scale);

  phi::funcs::ForRange<Context> for_range(dev_ctx, x.numel());
  phi::GetIntermediateParams<T> get_intermediate_params = {
    x.data<T>(),
    alpha.data<T>(),
    beta.data<T>(),
    gradient_scale.data<T>(),
    Qn,
    Qp,
    alpha_matrix.data<T>(),
    beta_matrix.data<T>(),
    mask.data<T>()};

  for_range(get_intermediate_params);

  // 设置求和范围为所有维度
  std::vector<int> v_dims(x.dims().size());
  std::iota(v_dims.begin(), v_dims.end(), 0);
  IntArray v_axes(v_dims);

  // alpha梯度
  SumKernel<T, Context>(
      dev_ctx, alpha_matrix, v_axes, x.dtype(), 0, alpha_grad);
  alpha_grad->Resize(alpha.dims());

  // beta梯度
  SumKernel<T, Context>(
      dev_ctx, beta_matrix, v_axes, x.dtype(), 0, beta_grad);
  beta_grad->Resize(beta.dims());

  // 输入梯度
  MultiplyKernel<T, Context>(dev_ctx, mask, out_grad, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(lsqplus_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LsqplusGradKernel,
                   float) {}
