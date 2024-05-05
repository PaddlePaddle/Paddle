// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/eigvals_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"

namespace phi {

template <typename T, typename enable = void>
struct PaddleComplex;

template <typename T>
struct PaddleComplex<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using type = dtype::complex<T>;
};

template <typename T>
struct PaddleComplex<
    T,
    typename std::enable_if<
        std::is_same<T, dtype::complex<float>>::value ||
        std::is_same<T, dtype::complex<double>>::value>::type> {
  using type = T;
};

template <typename T>
using PaddleCType = typename PaddleComplex<T>::type;
template <typename T>
using Real = typename dtype::Real<T>;

inline void CheckLapackEigResult(const int info, const std::string& name) {
  PADDLE_ENFORCE_LE(
      info,
      0,
      errors::PreconditionNotMet("The QR algorithm failed to compute all the "
                                 "eigenvalues in function %s.",
                                 name.c_str()));
  PADDLE_ENFORCE_GE(
      info,
      0,
      errors::InvalidArgument(
          "The %d-th argument has an illegal value in function %s.",
          -info,
          name.c_str()));
}

template <typename T, typename Context>
typename std::enable_if<std::is_floating_point<T>::value>::type LapackEigvals(
    const Context& ctx,
    const DenseTensor& input,
    DenseTensor* output,
    DenseTensor* work,
    DenseTensor* rwork /*unused*/) {
  DenseTensor a;  // will be overwritten when lapackEig exit
  Copy(ctx, input, input.place(), /*blocking=*/true, &a);

  DenseTensor w;
  int64_t n_dim = input.dims()[1];
  w.Resize(common::make_ddim({n_dim << 1}));
  T* w_data = ctx.template Alloc<T>(&w);

  int64_t work_mem = static_cast<int64_t>(work->memory_size());
  int64_t required_work_mem = 3 * n_dim * sizeof(T);
  PADDLE_ENFORCE_GE(
      work_mem,
      3 * n_dim * sizeof(T),
      errors::InvalidArgument(
          "The memory size of the work tensor in LapackEigvals function "
          "should be at least %" PRId64 " bytes, "
          "but received work\'s memory size = %" PRId64 " bytes.",
          required_work_mem,
          work_mem));

  int info = 0;
  phi::funcs::lapackEig<T>('N',
                           'N',
                           static_cast<int>(n_dim),
                           a.template data<T>(),
                           static_cast<int>(n_dim),
                           w_data,
                           nullptr,
                           1,
                           nullptr,
                           1,
                           work->template data<T>(),
                           static_cast<int>(work_mem / sizeof(T)),
                           static_cast<T*>(nullptr),
                           &info);

  std::string name = "phi::backend::dynload::dgeev_";
  if (input.dtype() == DataType::FLOAT64) {
    name = "phi::backend::dynload::sgeev_";
  }
  CheckLapackEigResult(info, name);

  funcs::ForRange<Context> for_range(ctx, n_dim);
  funcs::RealImagToComplexFunctor<PaddleCType<T>> functor(
      w_data, w_data + n_dim, output->template data<PaddleCType<T>>(), n_dim);
  for_range(functor);
}

template <typename T, typename Context>
typename std::enable_if<std::is_same<T, dtype::complex<float>>::value ||
                        std::is_same<T, dtype::complex<double>>::value>::type
LapackEigvals(const Context& ctx,
              const DenseTensor& input,
              DenseTensor* output,
              DenseTensor* work,
              DenseTensor* rwork) {
  DenseTensor a;  // will be overwritten when lapackEig exit
  Copy(ctx, input, input.place(), /*blocking=*/true, &a);

  int64_t work_mem = static_cast<int64_t>(work->memory_size());
  int64_t n_dim = input.dims()[1];
  int64_t required_work_mem = 3 * n_dim * sizeof(T);
  PADDLE_ENFORCE_GE(
      work_mem,
      3 * n_dim * sizeof(T),
      errors::InvalidArgument(
          "The memory size of the work tensor in LapackEigvals function "
          "should be at least %" PRId64 " bytes, "
          "but received work\'s memory size = %" PRId64 " bytes.",
          required_work_mem,
          work_mem));

  int64_t rwork_mem = static_cast<int64_t>(rwork->memory_size());
  int64_t required_rwork_mem = (n_dim << 1) * sizeof(dtype::Real<T>);
  PADDLE_ENFORCE_GE(
      rwork_mem,
      required_rwork_mem,
      errors::InvalidArgument(
          "The memory size of the rwork tensor in LapackEigvals function "
          "should be at least %" PRId64 " bytes, "
          "but received rwork\'s memory size = %" PRId64 " bytes.",
          required_rwork_mem,
          rwork_mem));

  int info = 0;
  phi::funcs::lapackEig<T, dtype::Real<T>>(
      'N',
      'N',
      static_cast<int>(n_dim),
      a.template data<T>(),
      static_cast<int>(n_dim),
      output->template data<T>(),
      nullptr,
      1,
      nullptr,
      1,
      work->template data<T>(),
      static_cast<int>(work_mem / sizeof(T)),
      rwork->template data<dtype::Real<T>>(),
      &info);

  std::string name = "phi::backend::dynload::cgeev_";
  if (input.dtype() == DataType::COMPLEX128) {
    name = "phi::backend::dynload::zgeev_";
  }
  CheckLapackEigResult(info, name);
}

void SpiltBatchSquareMatrix(const DenseTensor& input,
                            std::vector<DenseTensor>* output) {
  DDim input_dims = input.dims();
  int last_dim = input_dims.size() - 1;
  int n_dim = static_cast<int>(input_dims[last_dim]);

  DDim flattened_input_dims, flattened_output_dims;
  if (input_dims.size() > 2) {
    flattened_input_dims =
        common::flatten_to_3d(input_dims, last_dim - 1, last_dim);
  } else {
    flattened_input_dims = common::make_ddim({1, n_dim, n_dim});
  }

  DenseTensor flattened_input;
  flattened_input.ShareDataWith(input);
  flattened_input.Resize(flattened_input_dims);
  (*output) = flattened_input.Split(1, 0);
}

template <typename T, typename Context>
void EigvalsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  ctx.template Alloc<PaddleCType<T>>(out);

  std::vector<DenseTensor> x_matrices;
  SpiltBatchSquareMatrix(x, /*->*/ &x_matrices);

  int64_t n_dim = x_matrices[0].dims()[1];
  int64_t n_batch = static_cast<int64_t>(x_matrices.size());
  DDim out_dims = out->dims();
  out->Resize(common::make_ddim({n_batch, n_dim}));
  std::vector<DenseTensor> out_vectors = out->Split(1, 0);

  // query workspace size
  T qwork = T();
  int info = 0;
  funcs::lapackEig<T, dtype::Real<T>>('N',
                                      'N',
                                      static_cast<int>(n_dim),
                                      x_matrices[0].template data<T>(),
                                      static_cast<int>(n_dim),
                                      nullptr,
                                      nullptr,
                                      1,
                                      nullptr,
                                      1,
                                      &qwork,
                                      -1,
                                      static_cast<dtype::Real<T>*>(nullptr),
                                      &info);
  int64_t lwork = static_cast<int64_t>(qwork);

  DenseTensor work, rwork;

  work.Resize(common::make_ddim({lwork}));
  ctx.template Alloc<T>(&work);

  if (IsComplexType(x.dtype())) {
    rwork.Resize(common::make_ddim({n_dim << 1}));
    ctx.template Alloc<dtype::Real<T>>(&rwork);
  }

  for (int64_t i = 0; i < n_batch; ++i) {
    LapackEigvals<T, Context>(
        ctx, x_matrices[i], &out_vectors[i], &work, &rwork);
  }
  out->Resize(out_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(eigvals,
                   CPU,
                   ALL_LAYOUT,
                   phi::EigvalsKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
