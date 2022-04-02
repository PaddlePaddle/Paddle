// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T, typename enable = void>
struct PaddleComplex;

template <typename T>
struct PaddleComplex<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using type = paddle::platform::complex<T>;
};
template <typename T>
struct PaddleComplex<
    T, typename std::enable_if<
           std::is_same<T, platform::complex<float>>::value ||
           std::is_same<T, platform::complex<double>>::value>::type> {
  using type = T;
};

template <typename T>
using PaddleCType = typename PaddleComplex<T>::type;
template <typename T>
using Real = typename phi::dtype::Real<T>;

static void SpiltBatchSquareMatrix(const Tensor& input,
                                   std::vector<Tensor>* output) {
  DDim input_dims = input.dims();
  int last_dim = input_dims.size() - 1;
  int n_dim = input_dims[last_dim];

  DDim flattened_input_dims, flattened_output_dims;
  if (input_dims.size() > 2) {
    flattened_input_dims =
        phi::flatten_to_3d(input_dims, last_dim - 1, last_dim);
  } else {
    flattened_input_dims = phi::make_ddim({1, n_dim, n_dim});
  }

  Tensor flattened_input;
  flattened_input.ShareDataWith(input);
  flattened_input.Resize(flattened_input_dims);
  (*output) = flattened_input.Split(1, 0);
}

static void CheckLapackEigResult(const int info, const std::string& name) {
  PADDLE_ENFORCE_LE(info, 0, platform::errors::PreconditionNotMet(
                                 "The QR algorithm failed to compute all the "
                                 "eigenvalues in function %s.",
                                 name.c_str()));
  PADDLE_ENFORCE_GE(
      info, 0, platform::errors::InvalidArgument(
                   "The %d-th argument has an illegal value in function %s.",
                   -info, name.c_str()));
}

template <typename DeviceContext, typename T>
static typename std::enable_if<std::is_floating_point<T>::value>::type
LapackEigvals(const framework::ExecutionContext& ctx, const Tensor& input,
              Tensor* output, Tensor* work, Tensor* rwork /*unused*/) {
  Tensor a;  // will be overwritten when lapackEig exit
  framework::TensorCopy(input, input.place(), &a);

  Tensor w;
  int64_t n_dim = input.dims()[1];
  auto* w_data =
      w.mutable_data<T>(phi::make_ddim({n_dim << 1}), ctx.GetPlace());

  int64_t work_mem = work->memory_size();
  int64_t required_work_mem = 3 * n_dim * sizeof(T);
  PADDLE_ENFORCE_GE(
      work_mem, 3 * n_dim * sizeof(T),
      platform::errors::InvalidArgument(
          "The memory size of the work tensor in LapackEigvals function "
          "should be at least %" PRId64 " bytes, "
          "but received work\'s memory size = %" PRId64 " bytes.",
          required_work_mem, work_mem));

  int info = 0;
  phi::funcs::lapackEig<T>('N', 'N', static_cast<int>(n_dim),
                           a.template data<T>(), static_cast<int>(n_dim),
                           w_data, NULL, 1, NULL, 1, work->template data<T>(),
                           static_cast<int>(work_mem / sizeof(T)),
                           static_cast<T*>(NULL), &info);

  std::string name = "framework::platform::dynload::dgeev_";
  if (framework::TransToProtoVarType(input.dtype()) ==
      framework::proto::VarType::FP64) {
    name = "framework::platform::dynload::sgeev_";
  }
  CheckLapackEigResult(info, name);

  platform::ForRange<DeviceContext> for_range(
      ctx.template device_context<DeviceContext>(), n_dim);
  phi::funcs::RealImagToComplexFunctor<PaddleCType<T>> functor(
      w_data, w_data + n_dim, output->template data<PaddleCType<T>>(), n_dim);
  for_range(functor);
}

template <typename DeviceContext, typename T>
typename std::enable_if<std::is_same<T, platform::complex<float>>::value ||
                        std::is_same<T, platform::complex<double>>::value>::type
LapackEigvals(const framework::ExecutionContext& ctx, const Tensor& input,
              Tensor* output, Tensor* work, Tensor* rwork) {
  Tensor a;  // will be overwritten when lapackEig exit
  framework::TensorCopy(input, input.place(), &a);

  int64_t work_mem = work->memory_size();
  int64_t n_dim = input.dims()[1];
  int64_t required_work_mem = 3 * n_dim * sizeof(T);
  PADDLE_ENFORCE_GE(
      work_mem, 3 * n_dim * sizeof(T),
      platform::errors::InvalidArgument(
          "The memory size of the work tensor in LapackEigvals function "
          "should be at least %" PRId64 " bytes, "
          "but received work\'s memory size = %" PRId64 " bytes.",
          required_work_mem, work_mem));

  int64_t rwork_mem = rwork->memory_size();
  int64_t required_rwork_mem = (n_dim << 1) * sizeof(phi::dtype::Real<T>);
  PADDLE_ENFORCE_GE(
      rwork_mem, required_rwork_mem,
      platform::errors::InvalidArgument(
          "The memory size of the rwork tensor in LapackEigvals function "
          "should be at least %" PRId64 " bytes, "
          "but received rwork\'s memory size = %" PRId64 " bytes.",
          required_rwork_mem, rwork_mem));

  int info = 0;
  phi::funcs::lapackEig<T, phi::dtype::Real<T>>(
      'N', 'N', static_cast<int>(n_dim), a.template data<T>(),
      static_cast<int>(n_dim), output->template data<T>(), NULL, 1, NULL, 1,
      work->template data<T>(), static_cast<int>(work_mem / sizeof(T)),
      rwork->template data<phi::dtype::Real<T>>(), &info);

  std::string name = "framework::platform::dynload::cgeev_";
  if (framework::TransToProtoVarType(input.dtype()) ==
      framework::proto::VarType::COMPLEX64) {
    name = "framework::platform::dynload::zgeev_";
  }
  CheckLapackEigResult(info, name);
}

template <typename DeviceContext, typename T>
class EigvalsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    output->mutable_data<PaddleCType<T>>(ctx.GetPlace());

    std::vector<Tensor> input_matrices;
    SpiltBatchSquareMatrix(*input, /*->*/ &input_matrices);

    int64_t n_dim = input_matrices[0].dims()[1];
    int64_t n_batch = input_matrices.size();
    DDim output_dims = output->dims();
    output->Resize(phi::make_ddim({n_batch, n_dim}));
    std::vector<Tensor> output_vectors = output->Split(1, 0);

    // query workspace size
    T qwork;
    int info;
    phi::funcs::lapackEig<T, phi::dtype::Real<T>>(
        'N', 'N', static_cast<int>(n_dim), input_matrices[0].template data<T>(),
        static_cast<int>(n_dim), NULL, NULL, 1, NULL, 1, &qwork, -1,
        static_cast<phi::dtype::Real<T>*>(NULL), &info);
    int64_t lwork = static_cast<int64_t>(qwork);

    Tensor work, rwork;
    try {
      work.mutable_data<T>(phi::make_ddim({lwork}), ctx.GetPlace());
    } catch (memory::allocation::BadAlloc&) {
      LOG(WARNING) << "Failed to allocate Lapack workspace with the optimal "
                   << "memory size = " << lwork * sizeof(T) << " bytes, "
                   << "try reallocating a smaller workspace with the minimum "
                   << "required size = " << 3 * n_dim * sizeof(T) << " bytes, "
                   << "this may lead to bad performance.";
      lwork = 3 * n_dim;
      work.mutable_data<T>(phi::make_ddim({lwork}), ctx.GetPlace());
    }
    if (framework::IsComplexType(
            framework::TransToProtoVarType(input->dtype()))) {
      rwork.mutable_data<phi::dtype::Real<T>>(phi::make_ddim({n_dim << 1}),
                                              ctx.GetPlace());
    }

    for (int64_t i = 0; i < n_batch; ++i) {
      LapackEigvals<DeviceContext, T>(ctx, input_matrices[i],
                                      &output_vectors[i], &work, &rwork);
    }
    output->Resize(output_dims);
  }
};
}  // namespace operators
}  // namespace paddle
