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

#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace {
template <typename T>
void get_topk_pos(const T* data, int length, int k, int* pos) {
  size_t real_k = k < length ? k : length;

  std::vector<T> v(data, data + length);

  std::vector<int> topk_pos;
  T min_val = -10000000.0;
  while (topk_pos.size() < real_k) {
    T max_val = min_val;
    int max_pos = -1;
    for (int i = 0; i < length; ++i) {
      if (v[i] > max_val) {
        max_pos = i;
        max_val = v[i];
      }
    }

    assert(max_pos >= 0);

    topk_pos.push_back(max_pos);
    v[max_pos] = min_val;
  }

  assert(topk_pos.size() > 0);
  while (topk_pos.size() < (size_t)k) {
    topk_pos.push_back(-1);
  }

  for (size_t i = 0; i < topk_pos.size(); ++i) {
    pos[i] = topk_pos[i];
  }
}
}  // namespace

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
class SequenceTopkPoolingKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto* pos = context.Output<Tensor>("pos");

    auto channel_num = context.Attr<int>("channel_num");
    auto topk = context.Attr<int>("topk");
    std::vector<int> vec_pos_shape;
    auto batch_size = in->lod()[0].size() - 1;
    vec_pos_shape.push_back(batch_size * channel_num * topk);
    pos->Resize({framework::make_ddim(vec_pos_shape)});
    auto pos_data = pos->mutable_data<int>(context.GetPlace());

    auto in_lod = in->lod()[0];

    framework::Vector<size_t> vec_out_lod;
    vec_out_lod.reserve(batch_size + 1);
    for (int i = 0; i <= batch_size; ++i) {
      vec_out_lod.push_back(i * channel_num * topk);
    }
    framework::LoD lod_temp;
    lod_temp.push_back(vec_out_lod);
    out->set_lod(lod_temp);

    auto in_data = in->data<T>();
    auto out_data = out->mutable_data<T>(context.GetPlace());

    for (int i = 0; i < batch_size; ++i) {
      int total_size = in_lod[i + 1] - in_lod[i];
      if (total_size % channel_num != 0) {
        LOG(ERROR) << "input cannot mod channel num";
      }

      int feature_num = total_size / channel_num;
      auto in_offset_data = in_data + in_lod[i];
      for (int j = 0; j < channel_num; ++j) {
        auto input_slice_data = in_offset_data + j * feature_num;
        auto pos_slice_data = pos_data + i * topk * channel_num + j * topk;
        auto out_slice_data = out_data + i * topk * channel_num + j * topk;

        get_topk_pos<T>(input_slice_data, feature_num, topk, pos_slice_data);
        for (int k = 0; k < topk; ++k) {
          if (pos_slice_data[k] == -1) {
            out_slice_data[k] = 0.0;
          } else {
            out_slice_data[k] = input_slice_data[pos_slice_data[k]];
          }
        }
      }
    }
  }
};

template <typename T>
class SequenceTopkPoolingGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_grad = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* pos_input = context.Input<Tensor>("pos");
    auto* real_input = context.Input<LoDTensor>("X");

    auto channel_num = context.Attr<int>("channel_num");
    auto topk = context.Attr<int>("topk");

    auto out_lod = real_input->lod();
    in_grad->set_lod(out_lod);

    in_grad->mutable_data<T>(context.GetPlace());
    auto pos_data = pos_input->data<int>();

    auto out_data = out_grad->data<T>();

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<paddle::platform::CPUDeviceContext, T> zero;
    zero(dev_ctx, in_grad, static_cast<T>(0.0));

    auto in_data = in_grad->data<T>();

    auto out_offset = out_lod[0];

    auto batch_size = real_input->lod()[0].size() - 1;
    for (int i = 0; i < batch_size; ++i) {
      auto in_offset_data = in_data + out_offset[i];
      int total_size = out_offset[i + 1] - out_offset[i];
      int feature_num = total_size / channel_num;

      for (int j = 0; j < channel_num; ++j) {
        auto in_slice_data = in_offset_data + j * feature_num;
        auto pos_slice_data = pos_data + i * channel_num * topk + j * topk;
        auto out_slice_data = out_data + i * channel_num * topk + j * topk;

        for (int k = 0; k < topk; ++k) {
          if (pos_slice_data[k] == -1) {
            continue;
          } else {
            // LOG(ERROR) << i << " " << j << " " << k << " " <<
            // pos_slice_data[k] << " " << out_slice_data[k];
            in_slice_data[pos_slice_data[k]] = out_slice_data[k];
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
