// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/reduce_kernel.h"

#include "paddle/phi/kernels/gpu/reduce_amin_amax_common.h"
#include "paddle/phi/kernels/reduce_amin_grad_kernel.h"
#include "paddle/phi/kernels/reduce_max_grad_kernel.h"
#include "paddle/phi/kernels/reduce_mean_grad_kernel.h"
#include "paddle/phi/kernels/reduce_min_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/reduce_grad.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void ReduceSumGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  // get reduce_dim for reduce_mean_grad
  int dim_size = x.dims().size();
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);

  auto update_dims = common::vectorize(x.dims());
  for (auto i : reduce_dims) {
    update_dims[i] = 1;
  }

  // make new tensor
  DenseTensor new_out_grad(out_grad.dtype());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(common::make_ddim(update_dims));

  // call ReduceGrad
  dev_ctx.Alloc(x_grad, x.dtype());
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  phi::ReduceGrad<kps::IdentityFunctor<T, MPType>>(
      dev_ctx,
      &new_out_grad,
      x_grad,
      x.dtype(),
      kps::IdentityFunctor<T, MPType>());
}

template <typename T, typename Context>
void ReduceMinGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, x.dtype());
  reduce_all = recompute_reduce_all(x, dims, reduce_all);

  // get reduce_dim
  int dim_size = x.dims().size();
  auto reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);
  auto update_dims = common::vectorize(x.dims());
  for (auto i : reduce_dims) {
    update_dims[i] = 1;
  }

  // make new tensor of out and out_grad
  phi::DenseTensor new_out(out.type());
  new_out.ShareDataWith(out);
  new_out.Resize(common::make_ddim(update_dims));

  phi::DenseTensor new_out_grad(out_grad.type());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(common::make_ddim(update_dims));

  // make equal_out
  phi::DenseTensor* equal_out = new phi::DenseTensor();
  equal_out->Resize(x.dims());
  dev_ctx.template Alloc<T>(equal_out);

  // compute
  // 1. equal_out = Equal(x, y)
  std::vector<const phi::DenseTensor*> equal_inputs = {&new_out, &x};
  std::vector<phi::DenseTensor*> equal_outputs = {equal_out};
  funcs::BroadcastKernel<T>(
      dev_ctx, equal_inputs, &equal_outputs, funcs::EqualFunctor<T>(), 0);

  // 2. dx = dout * 1
  phi::MultiplyKernel<T, Context>(dev_ctx, new_out_grad, *equal_out, x_grad);
  delete equal_out;
}

template <typename T, typename Context>
void ReduceMeanGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out_grad,
                          const IntArray& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  // get reduce_dim and reduce_num for reduce_mean_grad
  int dim_size = x.dims().size();
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);

  auto update_dims = common::vectorize(x.dims());
  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (x.dims())[i];
    update_dims[i] = 1;
  }

  // make new tensor
  DenseTensor new_out_grad(out_grad.dtype());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(common::make_ddim(update_dims));

  // call BroadcastKernel
  dev_ctx.Alloc(x_grad, x.dtype());
  std::vector<const DenseTensor*> inputs = {&new_out_grad};
  std::vector<DenseTensor*> outputs = {x_grad};

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, kps::DivideFunctor<T, MPType>(reduce_num), 0);
}

template <typename T, typename Context>
void ReduceMaxGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const IntArray& dims,
                         bool keep_dim,
                         bool reduce_all,
                         DenseTensor* x_grad) {
  dev_ctx.Alloc(x_grad, x.dtype());
  reduce_all = recompute_reduce_all(x, dims, reduce_all);

  // get reduce_dim
  int dim_size = x.dims().size();
  auto reduce_dims =
      funcs::details::GetReduceDim(dims.GetData(), dim_size, reduce_all);
  auto update_dims = common::vectorize(x.dims());
  for (auto i : reduce_dims) {
    update_dims[i] = 1;
  }

  // make new tensor of out and out_grad
  phi::DenseTensor new_out(out.type());
  new_out.ShareDataWith(out);
  new_out.Resize(common::make_ddim(update_dims));

  phi::DenseTensor new_out_grad(out_grad.type());
  new_out_grad.ShareDataWith(out_grad);
  new_out_grad.Resize(common::make_ddim(update_dims));

  // make equal_out
  phi::DenseTensor* equal_out = new phi::DenseTensor();
  equal_out->Resize(x.dims());
  dev_ctx.template Alloc<T>(equal_out);

  // compute
  // 1. equal_out = Equal(x, y)
  std::vector<const phi::DenseTensor*> equal_inputs = {&new_out, &x};
  std::vector<phi::DenseTensor*> equal_outputs = {equal_out};
  funcs::BroadcastKernel<T>(
      dev_ctx, equal_inputs, &equal_outputs, funcs::EqualFunctor<T>(), 0);

  // 2. dx = dout * 1
  phi::MultiplyKernel<T, Context>(dev_ctx, new_out_grad, *equal_out, x_grad);
  delete equal_out;
}

template <typename T, typename Context>
void ReduceAMinGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          const std::vector<int64_t>& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  ReduceCudaAMaxAMinGrad<T, Context>(
      dev_ctx, x, out, out_grad, dims, keep_dim, reduce_all, x_grad);
}

template <typename T, typename Context>
void ReduceAMaxGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& out,
                          const DenseTensor& out_grad,
                          const std::vector<int64_t>& dims,
                          bool keep_dim,
                          bool reduce_all,
                          DenseTensor* x_grad) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  ReduceCudaAMaxAMinGrad<T, Context>(
      dev_ctx, x, out, out_grad, dims, keep_dim, reduce_all, x_grad);
}

template <typename T, typename Context>
void ReduceKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int root,
                  int reduce_type,
                  DenseTensor* out) {
  PADDLE_ENFORCE_GT(x.numel(),
                    0,
                    common::errors::InvalidArgument(
                        "Tensor need be reduced must not empty."));
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::NCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  gpuStream_t stream = dev_ctx.stream();
  PADDLE_ENFORCE_NOT_NULL(stream,
                          errors::NotFound("Should initialize NCCL firstly."));

  ncclRedOp_t red_type = ncclSum;
  switch (static_cast<ReduceType>(reduce_type)) {
    case ReduceType::kRedSum:
      red_type = ncclSum;
      break;
    case ReduceType::kRedMax:
      red_type = ncclMax;
      break;
    case ReduceType::kRedMin:
      red_type = ncclMin;
      break;
    case ReduceType::kRedProd:
      red_type = ncclProd;
      break;
#if NCCL_VERSION_CODE >= 21000
    case ReduceType::kRedAvg:
      red_type = ncclAvg;
      break;
#endif
  }
  comm_ctx->Reduce(out, x, red_type, root, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with GPU."));
#endif
}

}  // namespace phi

#if NCCL_VERSION_CODE >= 21000
PD_REGISTER_KERNEL(reduce,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(reduce,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceKernel,
                   float,
                   double,
                   int,
                   bool,
                   int8_t,
                   uint8_t,
                   int64_t,
                   phi::dtype::float16) {}
#endif

PD_REGISTER_KERNEL(amax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceAMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(amin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceAMinGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(max_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceMaxGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(mean_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceMeanGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(min_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceMinGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(sum_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReduceSumGradKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
