#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

#define DEFINE_SPARSE_UNARY_KERNEL(dense_kernel_func)                    \
  namespace phi {                                                        \
  namespace sparse {                                                     \
                                                                         \
  template <typename T, typename Context>                                \
  void SparseCoo##dense_kernel_func(const Context& dev_ctx,              \
                                    const SparseCooTensor& x,            \
                                    SparseCooTensor* out) {              \
    DenseTensor non_zero_indices =                                       \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_indices());       \
    DenseTensor non_zero_elements =                                      \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());      \
    phi::Copy(dev_ctx,                                                   \
              x.non_zero_indices(),                                      \
              dev_ctx.GetPlace(),                                        \
              false,                                                     \
              &non_zero_indices);                                        \
    phi::dense_kernel_func<T, Context>(                                  \
        dev_ctx, x.non_zero_elements(), &non_zero_elements);             \
    out->SetMember(non_zero_indices, non_zero_elements, x.dims(), true); \
  }                                                                      \
                                                                         \
  template <typename T, typename Context>                                \
  void SparseCsr##dense_kernel_func(const Context& dev_ctx,              \
                                    const SparseCsrTensor& x,            \
                                    SparseCsrTensor* out) {              \
    DenseTensor non_zero_crows =                                         \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_crows());         \
    DenseTensor non_zero_cols =                                          \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_cols());          \
    DenseTensor non_zero_elements =                                      \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());      \
    phi::Copy(dev_ctx,                                                   \
              x.non_zero_crows(),                                        \
              dev_ctx.GetPlace(),                                        \
              false,                                                     \
              &non_zero_crows);                                          \
    phi::Copy(dev_ctx,                                                   \
              x.non_zero_cols(),                                         \
              dev_ctx.GetPlace(),                                        \
              false,                                                     \
              &non_zero_cols);                                           \
    phi::dense_kernel_func<T, Context>(                                  \
        dev_ctx, x.non_zero_elements(), &non_zero_elements);             \
    out->SetMember(                                                      \
        non_zero_crows, non_zero_cols, non_zero_elements, x.dims());     \
  }                                                                      \
  }                                                                      \
  }

#define DEFINE_SPARSE_UNARY_GRAD_KERNEL(dense_kernel_func)                  \
  namespace phi {                                                           \
  namespace sparse {                                                        \
                                                                            \
  template <typename T, typename Context>                                   \
  void SparseCoo##dense_kernel_func(const Context& dev_ctx,                 \
                                    const SparseCooTensor& x,               \
                                    const SparseCooTensor& out_grad,        \
                                    SparseCooTensor* x_grad) {              \
    DenseTensor non_zero_indices =                                          \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_indices());          \
    DenseTensor non_zero_elements =                                         \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());         \
    phi::Copy(dev_ctx,                                                      \
              x.non_zero_indices(),                                         \
              dev_ctx.GetPlace(),                                           \
              false,                                                        \
              &non_zero_indices);                                           \
    phi::dense_kernel_func<T, Context>(dev_ctx,                             \
                                       x.non_zero_elements(),               \
                                       out_grad.non_zero_elements(),        \
                                       &non_zero_elements);                 \
    x_grad->SetMember(non_zero_indices, non_zero_elements, x.dims(), true); \
  }                                                                         \
                                                                            \
  template <typename T, typename Context>                                   \
  void SparseCsr##dense_kernel_func(const Context& dev_ctx,                 \
                                    const SparseCsrTensor& x,               \
                                    const SparseCsrTensor& out_grad,        \
                                    SparseCsrTensor* out) {                 \
    DenseTensor non_zero_crows =                                            \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_crows());            \
    DenseTensor non_zero_cols =                                             \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_cols());             \
    DenseTensor non_zero_elements =                                         \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());         \
    phi::Copy(dev_ctx,                                                      \
              x.non_zero_crows(),                                           \
              dev_ctx.GetPlace(),                                           \
              false,                                                        \
              &non_zero_crows);                                             \
    phi::Copy(dev_ctx,                                                      \
              x.non_zero_cols(),                                            \
              dev_ctx.GetPlace(),                                           \
              false,                                                        \
              &non_zero_cols);                                              \
    phi::dense_kernel_func<T, Context>(dev_ctx,                             \
                                       x.non_zero_elements(),               \
                                       out_grad.non_zero_elements(),        \
                                       &non_zero_elements);                 \
    out->SetMember(                                                         \
        non_zero_crows, non_zero_cols, non_zero_elements, x.dims());        \
  }                                                                         \
  }                                                                         \
  }

#define REGISTER_CPU_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func) \
  PD_REGISTER_KERNEL(sparse_coo_##kernel_name,                           \
                     CPU,                                                \
                     ALL_LAYOUT,                                         \
                     phi::sparse::SparseCoo##dense_kernel_func,          \
                     float,                                              \
                     double) {                                           \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);       \
  }                                                                      \
  PD_REGISTER_KERNEL(sparse_csr_##kernel_name,                           \
                     CPU,                                                \
                     ALL_LAYOUT,                                         \
                     phi::sparse::SparseCsr##dense_kernel_func,          \
                     float,                                              \
                     double) {                                           \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);       \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#define REGISTER_GPU_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func) \
  PD_REGISTER_KERNEL(sparse_coo_##kernel_name,                           \
                     GPU,                                                \
                     ALL_LAYOUT,                                         \
                     phi::sparse::SparseCoo##dense_kernel_func,          \
                     float,                                              \
                     double,                                             \
                     phi::dtype::float16) {                              \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);       \
  }                                                                      \
                                                                         \
  PD_REGISTER_KERNEL(sparse_csr_##kernel_name,                           \
                     GPU,                                                \
                     ALL_LAYOUT,                                         \
                     phi::sparse::SparseCsr##dense_kernel_func,          \
                     float,                                              \
                     double,                                             \
                     phi::dtype::float16) {                              \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);       \
  }
#else
// This macro definition is empty when GPU is disabled
#define REGISTER_GPU_SPARSE_UNARY_KERNEL(sparse_kernel_name, dense_kernel_func)
#endif

#define REGISTER_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func) \
  REGISTER_CPU_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func)   \
  REGISTER_GPU_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func)

#define DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(kernel_name,       \
                                                dense_kernel_func) \
  DEFINE_SPARSE_UNARY_KERNEL(dense_kernel_func)                    \
  REGISTER_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func)

#define DEFINE_AND_REGISTER_SPARSE_UNARY_GRAD_KERNEL(kernel_name,       \
                                                     dense_kernel_func) \
  DEFINE_SPARSE_UNARY_GRAD_KERNEL(dense_kernel_func)                    \
  REGISTER_SPARSE_UNARY_KERNEL(kernel_name, dense_kernel_func)

