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

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename T>
using vector = framework::Vector<T>;

vector<size_t> repeat_lod(vector<size_t> data, vector<size_t> starts,
                          vector<size_t> times, bool is_first) {
  vector<size_t> result;
  result.push_back(data[0]);
  size_t p = 0, start = 0, end = 0;
  if (is_first == true) {
    for (size_t i = 0; i < times.size(); ++i) {
      result.push_back(data.back() + times[i] * (data[i + 1] - data[i]));
    }
  } else {
    for (size_t i = 0; i < times.size(); ++i) {
      while (starts[i] != data[p] && p < data.size()) {
        ++p;
      }
      start = p;
      while (starts[i + 1] != data[p] && p < data.size()) {
        ++p;
      }
      end = p + 1;
      for (size_t j = 0; j < times[i]; ++j) {
        for (size_t index = start; index < end - 1; ++index) {
          result.push_back(result.back() + data[index + 1] - data[index]);
        }
      }
    }
  }
  return result;
}

template <typename Place, typename T>
void repeat_data(const T* src, T* dst, size_t size, vector<size_t> starts,
                 vector<size_t> times, Place place) {
  const T* src_p = src;
  T* dst_p = dst;
  size_t count = 0;
  for (size_t i = 0; i < times.size(); ++i) {
    count = size * (starts[i + 1] - starts[i]);
    for (size_t j = 0; j < times[i]; ++j) {
      memory::Copy(place, dst_p, place, src_p, sizeof(T) * count);
      dst_p += count;
    }
    src_p += count;
  }
}

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
      vector<size_t> level;
      for (int i = 0; i < x->dims()[0] + 1; ++i) {
        level.push_back(i);
      }
      x_lod.push_back(level);
    } else {
      x_lod.insert(x_lod.begin(), x_lod[0]);
    }

    size_t repeat = static_cast<size_t>(context.Attr<int>("repeat"));
    vector<size_t> repeats;
    if (repeat != 0) {
      for (int i = 0; i < x_lod[0].size() - 1; ++i) {
        repeats.push_back(repeat);
      }
      std::vector<int64_t> dims = framework::vectorize(x->dims());
      dims[0] = dims[0] * repeat;
      auto out_dims = framework::make_ddim(dims);
      out->Resize(out_dims);
    } else {
      auto* y = context.Input<LoDTensor>("Y");
      auto y_lod = y->lod();
      for (int i = 0; i < y_lod[0].size() - 1; ++i) {
        repeats.push_back((y_lod[0][i + 1] - y_lod[0][i]) /
                          (x_lod[0][i + 1] - x_lod[0][i]));
      }
      out->Resize(x_dims);
    }

    framework::LoD out_lod;
    auto level0 = repeat_lod(x_lod[0], x_lod[0], repeats, true);
    out_lod.push_back(level0);
    for (int i = 1; i < x_lod.size(); ++i) {
      out_lod.push_back(repeat_lod(x_lod[i], x_lod[0], repeats, false));
    }

    size_t element_len = framework::product(x_dims) / x_dims[0];
    T* out_data = out->mutable_data<T>(context.GetPlace());
    Place place = boost::get<Place>(context.GetPlace());
    repeat_data<Place, T>(x_data, out_data, element_len, x_lod[0], repeats,
                          place);
    out->set_lod(out_lod);
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
