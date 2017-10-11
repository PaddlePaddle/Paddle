/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "hl_cuda.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename Place, typename T>
class SeqExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());
    size_t repeat = static_cast<size_t>(context.Attr<int>("repeat"));

    if (repeat != 0) {
      if (x->lod().size() == 0) {
        std::vector<size_t> level0;
        for (size_t i = 0; i <= x->dims()[0]; i++) {
          level0.push_back(i * repeat);
        }
        framework::LoD out_lod;
        out_lod.push_back(level0);
        out->set_lod(out_lod);
      }
    }
    auto out_dim = out->dims();
    size_t element_len = framework::product(out_dim) / out_dim[0];
    std::vector<int> cpy_map(out_dim[0]);
    if (x->lod().size() == 0) {
      auto lod = out->lod();
      for (int i = 0; i < lod.size() - 1; ++i) {
        for (int j = lod[0][i]; i < lod[0][i + 1]; ++j) {
          cpy_map[j] = i;
        }
      }
    }
    if (platform::is_cpu_place(context.GetPlace())) {
      for (int i = 0; i < out_dim[0]; ++i) {
        memcpy(out_data + element_len * i, x_data + element_len * cpy_map[i],
               sizeof(T) * element_len);
      }
    } else {
      for (int i = 0; i < out_dim[0]; ++i) {
        hl_memcpy(out_data + element_len * i,
                  const_cast<T*>(x_data) + element_len * cpy_map[i],
                  sizeof(T) * element_len);
      }
    }
  }
};

template <typename Place, typename T>
class SeqExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    // auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    // d_x->mutable_data<T>(context.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle
