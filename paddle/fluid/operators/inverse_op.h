/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class InverseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
#ifdef PADDLE_WITH_MKLML
    auto* input = context.Input<framework::Tensor>("Input");
    auto* output = context.Output<framework::Tensor>("Output");

    const auto& input_dims = input->dims();
    const int rank = input_dims.size();
    int N = input_dims[rank - 1];
    int batch_size = rank > 2 ? input->numel() / (N * N) : 1;

    framework::Tensor ipiv;
    int* ipiv_ptr = ipiv.mutable_data<int>({N}, context.GetPlace());

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    T* output_ptr = output->mutable_data<T>(context.GetPlace());

    if (input->data<T>() != output_ptr) {
      framework::TensorCopy(*input, context.GetPlace(), output);
    }

    for (int i = 0; i < batch_size; ++i) {
      T* A = output_ptr + i * N * N;

      // Compute the LU Factorization of a general m-by-n matrix: A = P*L*U
      blas.GETRF(N, N, A, ipiv_ptr);

      // Computes the inverse of an LU-factored general matrix.
      blas.GETRI(N, A, ipiv_ptr);
    }
#else
    PADDLE_THROW(
        platform::errors::Unimplemented("The CPU kernel of matrix's inverse "
                                        "without MKLML is not implemented."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
