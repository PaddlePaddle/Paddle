/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using platform::MKLDNNDeviceContext;
using framework::ExecutionContext;
using Tensor = framework::Tensor;

template <typename T>
class MatMulGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override;

 private:
  void ExecuteMatMulGrad(const ExecutionContext& ctx,
                         const MKLDNNDeviceContext& dev_ctx,
                         const dnnl::engine& engine, Tensor* x, bool trans_x,
                         bool is_fold_init_dims_x, Tensor* y, bool trans_y,
                         bool is_fold_init_dims_y, Tensor* out) const;
  void RunKernel(const ExecutionContext& ctx) const;
};
}  // namespace operators
}  // namespace paddle
