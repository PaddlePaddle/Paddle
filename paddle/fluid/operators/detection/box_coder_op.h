/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

enum class BoxCodeType { kEncodeCenterSize = 0, kDecodeCenterSize = 1 };

inline BoxCodeType GetBoxCodeType(const std::string& type) {
  if (type == "encode_center_size") {
    return BoxCodeType::kEncodeCenterSize;
  } else if (type == "decode_center_size") {
    return BoxCodeType::kDecodeCenterSize;
  }
  PADDLE_THROW("Not support type %s.", type);
}

template <typename DeviceContext, typename T>
class BoxCoderKernel : public framework::OpKernel<T> {
 public:
  void EncodeCenterSize(const framework::Tensor* target_box,
                        const framework::Tensor* prior_box,
                        const framework::Tensor* prior_box_var,
                        const bool normalized, T* output) const {
    int64_t row = target_box->dims()[0];
    int64_t col = prior_box->dims()[0];
    int64_t len = prior_box->dims()[1];
    auto* target_box_data = target_box->data<T>();
    auto* prior_box_data = prior_box->data<T>();
    const T* prior_box_var_data = nullptr;
    if (prior_box_var) prior_box_var_data = prior_box_var->data<T>();

    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        T prior_box_width = prior_box_data[j * len + 2] -
                            prior_box_data[j * len] + (normalized == false);
        T prior_box_height = prior_box_data[j * len + 3] -
                             prior_box_data[j * len + 1] +
                             (normalized == false);
        T prior_box_center_x =
            (prior_box_data[j * len + 2] + prior_box_data[j * len]) / 2;
        T prior_box_center_y =
            (prior_box_data[j * len + 3] + prior_box_data[j * len + 1]) / 2;

        T target_box_center_x =
            (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
        T target_box_center_y =
            (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
        T target_box_width = target_box_data[i * len + 2] -
                             target_box_data[i * len] + (normalized == false);
        T target_box_height = target_box_data[i * len + 3] -
                              target_box_data[i * len + 1] +
                              (normalized == false);

        size_t offset = i * col * len + j * len;
        output[offset] =
            (target_box_center_x - prior_box_center_x) / prior_box_width;
        output[offset + 1] =
            (target_box_center_y - prior_box_center_y) / prior_box_height;
        output[offset + 2] =
            std::log(std::fabs(target_box_width / prior_box_width));
        output[offset + 3] =
            std::log(std::fabs(target_box_height / prior_box_height));
        if (prior_box_var) {
          output[offset] /= prior_box_var_data[j * len];
          output[offset + 1] /= prior_box_var_data[j * len + 1];
          output[offset + 2] /= prior_box_var_data[j * len + 2];
          output[offset + 3] /= prior_box_var_data[j * len + 3];
        }
      }
    }
  }
  void DecodeCenterSize(const framework::Tensor* target_box,
                        const framework::Tensor* prior_box,
                        const framework::Tensor* prior_box_var,
                        const bool normalized, T* output) const {
    int64_t row = target_box->dims()[0];
    int64_t col = prior_box->dims()[0];
    int64_t len = prior_box->dims()[1];

    auto* target_box_data = target_box->data<T>();
    auto* prior_box_data = prior_box->data<T>();
    const T* prior_box_var_data = nullptr;
    if (prior_box_var) prior_box_var_data = prior_box_var->data<T>();

    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        size_t offset = i * col * len + j * len;
        T prior_box_width = prior_box_data[j * len + 2] -
                            prior_box_data[j * len] + (normalized == false);
        T prior_box_height = prior_box_data[j * len + 3] -
                             prior_box_data[j * len + 1] +
                             (normalized == false);
        T prior_box_center_x =
            (prior_box_data[j * len + 2] + prior_box_data[j * len]) / 2;
        T prior_box_center_y =
            (prior_box_data[j * len + 3] + prior_box_data[j * len + 1]) / 2;

        T target_box_center_x = 0, target_box_center_y = 0;
        T target_box_width = 0, target_box_height = 0;
        if (prior_box_var) {
          target_box_center_x = prior_box_var_data[j * len] *
                                    target_box_data[offset] * prior_box_width +
                                prior_box_center_x;
          target_box_center_y = prior_box_var_data[j * len + 1] *
                                    target_box_data[offset + 1] *
                                    prior_box_height +
                                prior_box_center_y;
          target_box_width = std::exp(prior_box_var_data[j * len + 2] *
                                      target_box_data[offset + 2]) *
                             prior_box_width;
          target_box_height = std::exp(prior_box_var_data[j * len + 3] *
                                       target_box_data[offset + 3]) *
                              prior_box_height;
        } else {
          target_box_center_x =
              target_box_data[offset] * prior_box_width + prior_box_center_x;
          target_box_center_y = target_box_data[offset + 1] * prior_box_height +
                                prior_box_center_y;
          target_box_width =
              std::exp(target_box_data[offset + 2]) * prior_box_width;
          target_box_height =
              std::exp(target_box_data[offset + 3]) * prior_box_height;
        }

        output[offset] = target_box_center_x - target_box_width / 2;
        output[offset + 1] = target_box_center_y - target_box_height / 2;
        output[offset + 2] =
            target_box_center_x + target_box_width / 2 - (normalized == false);
        output[offset + 3] =
            target_box_center_y + target_box_height / 2 - (normalized == false);
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* prior_box = context.Input<framework::Tensor>("PriorBox");
    auto* prior_box_var = context.Input<framework::Tensor>("PriorBoxVar");
    auto* target_box = context.Input<framework::LoDTensor>("TargetBox");
    auto* output_box = context.Output<framework::Tensor>("OutputBox");

    if (target_box->lod().size()) {
      PADDLE_ENFORCE_EQ(target_box->lod().size(), 1UL,
                        "Only support 1 level of LoD.");
    }
    auto row = target_box->dims()[0];
    auto col = prior_box->dims()[0];
    auto len = prior_box->dims()[1];

    output_box->mutable_data<T>({row, col, len}, context.GetPlace());

    auto code_type = GetBoxCodeType(context.Attr<std::string>("code_type"));
    bool normalized = context.Attr<bool>("box_normalized");
    T* output = output_box->data<T>();
    if (code_type == BoxCodeType::kEncodeCenterSize) {
      EncodeCenterSize(target_box, prior_box, prior_box_var, normalized,
                       output);
    } else if (code_type == BoxCodeType::kDecodeCenterSize) {
      DecodeCenterSize(target_box, prior_box, prior_box_var, normalized,
                       output);
    }
  }
};

}  // namespace operators
}  // namespace paddle
