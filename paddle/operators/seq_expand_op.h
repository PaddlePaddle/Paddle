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

#include "paddle/framework/op_registry.h"
#include "paddle/memory/memcpy.h"
#include "unsupported/Eigen/CXX11/Tensor"

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
    auto x_dims = x->dims();
    auto x_lod = x->lod();

    if (x_lod.size() == 0) {
      framework::Vector<size_t> level;
      for (int i = 0; i < x->dims()[0] + 1; ++i) {
        level.push_back(i);
      }
      x_lod.push_back(level);
    } else {
      x_lod.insert(x_lod.begin(), x_lod[0]);
    }

    size_t repeat = static_cast<size_t>(context.Attr<int>("repeat"));
    framework::Vector<size_t> scales;
    if (repeat != 0) {
      for (int i = 0; i < x_lod[0].size() - 1; ++i) {
        scales.push_back(repeat);
      }
      std::vector<int64_t> dims = framework::vectorize(x->dims());
      dims[0] = dims[0] * repeat;
      auto out_dims = framework::make_ddim(dims);
      out->Resize(out_dims);
    } else {
      auto* y = context.Input<LoDTensor>("Y");
      auto y_lod = y->lod();
      for (int i = 0; i < y_lod[0].size() - 1; ++i) {
        scales.push_back((y_lod[0][i + 1] - y_lod[0][i]) /
                         (x_lod[0][i + 1] - x_lod[0][i]));
      }
      out->Resize(y->dims());
    }

    framework::LoD out_lod;
    auto level0 = framework::expand_lod(x_lod[0], x_lod[0], scales, false);
    out_lod.push_back(level0);
    for (int i = 1; i < x_lod.size(); ++i) {
      out_lod.push_back(
          framework::expand_lod(x_lod[i], x_lod[0], scales, true));
    }

    size_t element_len = framework::product(x_dims) / x_dims[0];
    T* out_data = out->mutable_data<T>(context.GetPlace());

    // copy data
    Place place = boost::get<Place>(context.GetPlace());
    size_t count = 0;
    for (size_t i = 0; i < scales.size(); ++i) {
      count = element_len * (x_lod[0][i + 1] - x_lod[0][i]);
      for (size_t j = 0; j < scales[i]; ++j) {
        memory::Copy(place, out_data, place, x_data, sizeof(T) * count);
        out_data += count;
      }
      x_data += count;
    }

    out->set_lod(out_lod);
  }
};

template <typename Place, typename T>
class SeqExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x = context.Input<LoDTensor>("X");
    auto* out = context.Input<LoDTensor>("Out");
    auto* d_x = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto out_lod = out->lod();
    d_x->set_lod(x->lod());
    const T* d_out_data = d_out->data<T>();
    auto d_out_dims = d_out->dims();
    T* d_x_data = d_x->mutable_data<T>(context.GetPlace());
    size_t element_len = framework::product(d_out_dims) / d_out_dims[0];
    for (size_t i = 0; i < out->NumElements(); ++i) {
      size_t ele_count = out_lod[0][i + 1] - out_lod[0][i];
      size_t repeat = out->NumElements(0, i);
      Eigen::TensorMap<Eigen::Tensor<const T, 2>> d_out_t(
          d_out_data, static_cast<int>(repeat),
          static_cast<int>((ele_count * element_len) / repeat));
      Eigen::TensorMap<Eigen::Tensor<T, 1>> d_x_t(
          d_x_data, static_cast<int>((ele_count * element_len) / repeat));
      auto place = context.GetEigenDevice<Place>();
      d_x_t.device(place) = d_out_t.sum(Eigen::array<int, 1>({0}));
      d_out_data += (ele_count * element_len);
      d_x_data += ((ele_count * element_len) / repeat);
    }
  }
};

}  // namespace operators
}  // namespace paddle
