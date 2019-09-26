/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <limits>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

template <typename T>
void get_topk_pos(const T* data, int length, int k, int* pos) {
  size_t real_k = k < length ? k : length;

  std::vector<T> v(data, data + length);

  std::vector<int> topk_pos;
  T min_val = std::numeric_limits<T>::lowest();
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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceTopkAvgPoolingKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* row = context.Input<LoDTensor>("ROW");
    auto* col = context.Input<LoDTensor>("COLUMN");
    auto* out = context.Output<LoDTensor>("Out");
    auto* pos = context.Output<Tensor>("pos");

    PADDLE_ENFORCE_EQ(in->lod().empty(), false,
                      "Input(X) Tensor of SequenceTopkAvgPoolingOp does not "
                      "contain LoD information.");
    PADDLE_ENFORCE_EQ(row->lod().empty(), false,
                      "Input(ROW) Tensor of SequenceTopkAvgPoolingOp does not "
                      "contain LoD information.");
    PADDLE_ENFORCE_EQ(col->lod().empty(), false,
                      "Input(COLUMN) Tensor of SequenceTopkAvgPoolingOp does "
                      "not contain LoD information.");

    auto channel_num = context.Attr<int>("channel_num");
    auto topks = context.Attr<std::vector<int>>("topks");
    auto k_num = topks.size();
    auto max_k = topks[topks.size() - 1];
    std::vector<int> vec_pos_shape;
    auto in_lod = in->lod()[0];

    auto row_lod = row->lod()[0];
    auto col_lod = col->lod()[0];
    int batch_size = row_lod.size() - 1;
    int pos_total_size = row_lod[batch_size] * channel_num * max_k;
    vec_pos_shape.push_back(pos_total_size);
    pos->Resize({framework::make_ddim(vec_pos_shape)});
    auto pos_data = pos->mutable_data<int>(context.GetPlace());

    int offset = 0;
    framework::Vector<size_t> vec_out_lod;
    vec_out_lod.reserve(batch_size + 1);
    for (int i = 0; i <= batch_size; ++i) {
      offset = row_lod[i];
      vec_out_lod.push_back(offset);
    }

    framework::LoD lod_temp;
    lod_temp.push_back(vec_out_lod);
    out->set_lod(lod_temp);

    auto din_data = in->data<T>();
    auto dout_data = out->mutable_data<T>(context.GetPlace());

    T* sum_data = new T[max_k];
    for (int i = 0; i < batch_size; ++i) {
      int total_size = in_lod[i + 1] - in_lod[i];
      int row_size = row_lod[i + 1] - row_lod[i];
      int col_size = col_lod[i + 1] - col_lod[i];
      PADDLE_ENFORCE_EQ(total_size, channel_num * row_size * col_size,
                        "size wrong in sequence_topk_avg_pooling_op!");

      int feature_num = row_size * col_size;
      for (int j = 0; j < channel_num; ++j) {
        auto input_offset_feature_data = din_data + in_lod[i] + j * feature_num;

        for (int r = 0; r < row_size; ++r) {
          auto row_data = input_offset_feature_data + r * col_size;

          auto pos_slice_data = pos_data + row_lod[i] * channel_num * max_k +
                                r * channel_num * max_k + j * max_k;
          auto out_slice_data = dout_data + row_lod[i] * channel_num * k_num +
                                r * channel_num * k_num + j * k_num;

          get_topk_pos<T>(row_data, col_size, max_k, pos_slice_data);
          if (pos_slice_data[0] == -1) {
            sum_data[0] = 0.0;
          } else {
            sum_data[0] = row_data[pos_slice_data[0]];
          }
          for (int k = 1; k < max_k; ++k) {
            if (pos_slice_data[k] == -1) {
              sum_data[k] = sum_data[k - 1];
            } else {
              sum_data[k] = sum_data[k - 1] + row_data[pos_slice_data[k]];
            }
          }
          for (size_t k = 0; k < k_num; ++k) {
            out_slice_data[k] = sum_data[topks[k] - 1] / topks[k];
          }
        }
      }
    }
    delete[] sum_data;
  }
};

template <typename DeviceContext, typename T>
class SequenceTopkAvgPoolingGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_in = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* pos_input = context.Input<Tensor>("pos");
    auto* row_input = context.Input<LoDTensor>("ROW");
    auto* col_input = context.Input<LoDTensor>("COLUMN");
    auto* forward_input = context.Input<LoDTensor>("X");

    int batch_size = row_input->lod()[0].size() - 1;
    auto channel_num = context.Attr<int>("channel_num");
    auto topks = context.Attr<std::vector<int>>("topks");
    auto k_num = topks.size();
    auto max_k = topks[k_num - 1];

    auto out_lod = forward_input->lod();
    d_in->set_lod(out_lod);

    d_in->mutable_data<T>(context.GetPlace());
    auto pos_data = pos_input->data<int>();
    auto dout_data = d_out->data<T>();

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<paddle::platform::CPUDeviceContext, T> zero;
    zero(dev_ctx, d_in, static_cast<T>(0.0));

    auto din_data = d_in->data<T>();

    auto out_offset = out_lod[0];
    auto row_lod = row_input->lod()[0];
    auto col_lod = col_input->lod()[0];

    for (int i = 0; i < batch_size; ++i) {
      int row_size = row_lod[i + 1] - row_lod[i];
      int col_size = col_lod[i + 1] - col_lod[i];
      int feature_num = row_size * col_size;

      for (int j = 0; j < channel_num; ++j) {
        auto in_offset_feature_data =
            din_data + out_offset[i] + j * feature_num;

        for (int r = 0; r < row_size; r++) {
          auto row_data = dout_data + row_lod[i] * channel_num * k_num +
                          r * channel_num * k_num + j * k_num;
          auto pos_slice_data = pos_data + row_lod[i] * channel_num * max_k +
                                r * channel_num * max_k + j * max_k;
          auto in_slice_data = in_offset_feature_data + r * col_size;

          for (size_t m = 0; m < k_num; ++m) {
            for (int k = 0; k < topks[m]; ++k) {
              if (pos_slice_data[k] == -1) {
                break;
              } else {
                in_slice_data[pos_slice_data[k]] += row_data[m] / topks[m];
              }
            }
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
