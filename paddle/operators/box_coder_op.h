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
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

enum class BoxCodeType { kEncodeCenterSize = 0, kDecodeCenterSize = 1 };

inline BoxCodeType GetBoxCodeType(const std::string& type) {
  if (type == "encode_center_size") {
    return BoxCodeType::kEncodeCenterSize;
  } else if (type == "decode_center_size") {
    return BoxCodeType::kDecodeCenterSize;
  }
  PADDLE_THROW("Not support type %s.", type);
}

template <typename T>
class BoxCoderKernel : public framework::OpKernel<T> {
 public:
  void EncodeCenterSize(const Tensor& target_box, const Tensor& prior_box,
                        const Tensor& prior_box_var, T* output) const {
    PADDLE_ENFORCE_EQ(target_box.dims().size(), 2,
                      "The rank of target_box must be 2.");
    PADDLE_ENFORCE_EQ(prior_box.dims().size(), 2,
                      "The rank of prior_box must be 2.");
    PADDLE_ENFORCE_EQ(prior_box_var.dims().size(), 2,
                      "The rank of prior_box_var must be 2.");
    PADDLE_ENFORCE_EQ(prior_box.dims()[0], prior_box_var.dims()[0],
                      "The dims of prior_box must equal to prior_box_var.");

    int64_t row = target_box.dims()[0];
    int64_t col = prior_box.dims()[0];
    auto* target_box_data = target_box.data<T>();
    auto* prior_box_data = prior_box.data<T>();
    auto* prior_box_var_data = prior_box_var.data<T>();

    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        T prior_box_width = prior_box_data[j * 4 + 2] - prior_box_data[j * 4];
        T prior_box_height =
            prior_box_data[j * 4 + 3] - prior_box_data[j * 4 + 1];
        T prior_box_center_x =
            (prior_box_data[j * 4 + 2] + prior_box_data[j * 4]) / 2;
        T prior_box_center_y =
            (prior_box_data[j * 4 + 3] + prior_box_data[j * 4 + 1]) / 2;

        T target_box_center_x =
            (target_box_data[i * 4 + 2] + target_box_data[i * 4]) / 2;
        T target_box_center_y =
            (target_box_data[i * 4 + 3] + target_box_data[i * 4 + 1]) / 2;
        T target_box_width =
            target_box_data[i * 4 + 2] - target_box_data[i * 4];
        T target_box_height =
            target_box_data[i * 4 + 3] - target_box_data[i * 4 + 1];

        size_t offset = i * col * 4 + j * 4;
        output[offset] = (target_box_center_x - prior_box_center_x) /
                         prior_box_width / prior_box_var_data[j * 4];
        output[offset + 1] = (target_box_center_y - prior_box_center_y) /
                             prior_box_height / prior_box_var_data[j * 4 + 1];
        output[offset + 2] =
            std::log(std::fabs(target_box_width / prior_box_width)) /
            prior_box_var_data[j * 4 + 2];
        output[offset + 3] =
            std::log(std::fabs(target_box_height / prior_box_height)) /
            prior_box_var_data[j * 4 + 3];
      }
    }
  }
  void DecodeCenterSize(const Tensor& target_box, const Tensor& prior_box,
                        const Tensor& prior_box_var, T* output) const {
    PADDLE_ENFORCE_EQ(target_box.dims().size(), 2,
                      "The rank of target_box must be 2.");
    PADDLE_ENFORCE_EQ(prior_box.dims().size(), 2,
                      "The rank of prior_box must be 2.");
    PADDLE_ENFORCE_EQ(prior_box_var.dims().size(), 2,
                      "The rank of prior_box_var must be 2.");
    PADDLE_ENFORCE_EQ(prior_box.dims()[0], prior_box_var.dims()[0],
                      "The dims of prior_box must equal to prior_box_var.");

    int64_t row = target_box.dims()[0];
    int64_t col = prior_box.dims()[0];

    auto* target_box_data = target_box.data<T>();
    auto* prior_box_data = prior_box.data<T>();
    auto* prior_box_var_data = prior_box_var.data<T>();

    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        T prior_box_width = prior_box_data[j * 4 + 2] - prior_box_data[j * 4];
        T prior_box_height =
            prior_box_data[j * 4 + 3] - prior_box_data[j * 4 + 1];
        T prior_box_center_x =
            (prior_box_data[j * 4 + 2] + prior_box_data[j * 4]) / 2;
        T prior_box_center_y =
            (prior_box_data[j * 4 + 3] + prior_box_data[j * 4 + 1]) / 2;

        T target_box_center_x = prior_box_var_data[j * 4] *
                                    target_box_data[i * 4] * prior_box_width +
                                prior_box_center_x;
        T target_box_center_y = prior_box_var_data[j * 4 + 1] *
                                    target_box_data[i * 4 + 1] *
                                    prior_box_height +
                                prior_box_center_y;
        T target_box_width = std::exp(prior_box_var_data[j * 4 + 2] *
                                      target_box_data[i * 4 + 2]) *
                             prior_box_width;
        T target_box_height = std::exp(prior_box_var_data[j * 4 + 3] *
                                       target_box_data[i * 4 + 3]) *
                              prior_box_height;

        size_t offset = i * col * 4 + j * 4;
        output[offset] = target_box_center_x - target_box_width / 2;
        output[offset + 1] = target_box_center_y - target_box_height / 2;
        output[offset + 2] = target_box_center_x + target_box_width / 2;
        output[offset + 3] = target_box_center_y + target_box_height / 2;
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* prior_box = context.Input<framework::Tensor>("PriorBox");
    auto* prior_box_var = context.Input<framework::Tensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* output_box = context.Output<Tensor>("OutputBox");

    if (target_box->lod().size()) {
      PADDLE_ENFORCE_EQ(target_box->lod().size(), 1UL,
                        "Only support 1 level of LoD.");
    }
    auto row = target_box->dims()[0];
    auto col = prior_box->dims()[0];

    output_box->mutable_data<T>({row, col, 4}, context.GetPlace());

    auto code_type = GetBoxCodeType(context.Attr<std::string>("code_type"));
    T* output = output_box->data<T>();
    if (code_type == BoxCodeType::kEncodeCenterSize) {
      EncodeCenterSize(*target_box, *prior_box, *prior_box_var, output);
    } else if (code_type == BoxCodeType::kDecodeCenterSize) {
      DecodeCenterSize(*target_box, *prior_box, *prior_box_var, output);
    }
  }
};

}  // namespace operators
}  // namespace paddle
