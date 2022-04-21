#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/sparse/utils.h"

DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(sin, SinKernel)

// NOTE: the following code is to bypass the restriction of Paddle
// kernel registration mechanism. Do NOT refactor them unless you
// know what you are doing.
// If you want to implement any new kernel, please follow the above
// `log`, do NOT follow the following `sqrt`.
DEFINE_SPARSE_UNARY_KERNEL(SqrtKernel)

PD_REGISTER_KERNEL(sparse_coo_sqrt,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooSqrtKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_REGISTER_KERNEL(sparse_csr_sqrt,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrSqrtKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(sparse_coo_sqrt,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooSqrtKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_csr_sqrt,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrSqrtKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
#endif
