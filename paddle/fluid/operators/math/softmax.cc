/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct SequenceSoftmaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const LoDTensor& x,
                  LoDTensor* out) {
    auto& lod = x.lod();
    const size_t level = lod.size() - 1;
    for (int i = 0; i < static_cast<int>(lod[level].size()) - 1; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);
      Tensor x_i = x.Slice(start_pos, end_pos);
      Tensor out_i = out->Slice(start_pos, end_pos);

      // Reshape from (end_pos - start_pos) x 1UL to 1UL x (end_pos - start_pos)
      framework::DDim dims_i = framework::make_ddim({1UL, end_pos - start_pos});
      x_i.Resize(dims_i);
      out_i.Resize(dims_i);
      math::SoftmaxFunctor<platform::CPUDeviceContext, T>()(ctx, &x_i, &out_i);
    }
  }
};

template <typename T>
struct SequenceSoftmaxGradFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::LoDTensor& out,
                  const framework::LoDTensor& dout, framework::LoDTensor* dx) {
    auto lod = dx->lod();
    const size_t level = lod.size() - 1;

    for (int i = 0; i < static_cast<int>(lod[level].size()) - 1; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);

      Tensor out_i = out.Slice(start_pos, end_pos);
      Tensor out_grad_i = dout.Slice(start_pos, end_pos);
      Tensor x_grad_i = dx->Slice(start_pos, end_pos);

      // Reshape from (end_pos - start_pos) x 1UL to 1UL x (end_pos - start_pos)
      framework::DDim dims_i = framework::make_ddim({1UL, end_pos - start_pos});
      out_i.Resize(dims_i);
      out_grad_i.Resize(dims_i);
      x_grad_i.Resize(dims_i);
      math::SoftmaxGradFunctor<platform::CPUDeviceContext, T>()(
          ctx, &out_i, &out_grad_i, &x_grad_i);
    }
  }
};

template class SoftmaxFunctor<platform::CPUDeviceContext, float>;
template class SoftmaxFunctor<platform::CPUDeviceContext, double>;
template class SoftmaxGradFunctor<platform::CPUDeviceContext, float>;
template class SoftmaxGradFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
