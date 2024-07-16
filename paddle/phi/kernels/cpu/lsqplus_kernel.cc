#include "paddle/phi/kernels/lsqplus_kernel.h"

namespace phi {
template <typename T, typename Context>
void LsqplusKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& alpha,
    const DenseTensor& beta,
    const DenseTensor& g,
    int Qn,
    int Qp,
    DenseTensor* out) {
    
    phi::funcs::ForRange<Context> for_range(dev_ctx, x.numel());
    phi::LsqplusFakequant<T> fake_quant = {
        x.data<T>(),
        alpha.data<T>(),
        beta.data<T>(),
        Qn,
        Qp,
        dev_ctx.template Alloc<T>(out)
    };
    for_range(fake_quant);
}

} // namespace phi

PD_REGISTER_KERNEL(lsqplus,
                   CPU,
                   ALL_LAYOUT,
                   phi::LsqplusKernel,
                   float) {}
