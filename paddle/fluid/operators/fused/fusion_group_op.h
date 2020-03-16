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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_code.h"

namespace paddle {
namespace operators {

static void MutableMultiTypeData(
    std::vector<paddle::framework::LoDTensor*>* var,
    const std::vector<std::string>& data_type, const platform::Place& place) {
  for (size_t i = 0; i < var->size(); i++) {
    if (data_type[i] == "float") {
      (*var)[i]->mutable_data<float>(place);
    } else if (data_type[i] == "double") {
      (*var)[i]->mutable_data<double>(place);
    } else if (data_type[i] == "::paddle::platform::float16") {
      (*var)[i]->mutable_data<paddle::platform::float16>(place);
    }
  }
}

template <typename DeviceContext, typename T>
class FusionGroupKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("Inputs");
    auto outs = ctx.MultiOutput<framework::LoDTensor>("Outs");
    int type = ctx.Attr<int>("type");
    auto outs_type = ctx.Attr<std::vector<std::string>>("outs_data_type");
    auto inputs_type = ctx.Attr<std::vector<std::string>>("inputs_data_type");

    size_t num_ins = ins.size();
    size_t num_outs = outs.size();

    auto place = ctx.GetPlace();

    MutableMultiTypeData(&outs, outs_type, place);

    std::string func_name = ctx.Attr<std::string>("func_name");
    platform::DeviceCode* dev_code =
        platform::DeviceCodePool::Instance().Get(place, func_name);
    VLOG(3) << "func_name: " << func_name;

    if (type == 0) {
      size_t n = ins[0]->numel();
      std::vector<void*> args;
      args.push_back(&n);
      std::vector<const void*> ptrs(num_ins + num_outs);
      for (size_t i = 0; i < num_ins; ++i) {
        if (inputs_type[i] == "::paddle::platform::float16") {
          ptrs[i] = ins[i]->data<paddle::platform::float16>();
        } else if (inputs_type[i] == "double") {
          ptrs[i] = ins[i]->data<double>();
        } else if (inputs_type[i] == "float") {
          ptrs[i] = ins[i]->data<float>();
        }
        args.push_back(&ptrs[i]);
      }
      for (size_t j = 0; j < num_outs; ++j) {
        if (outs_type[j] == "::paddle::platform::float16") {
          ptrs[num_ins + j] = outs[j]->data<paddle::platform::float16>();
        } else if (outs_type[j] == "double") {
          ptrs[num_ins + j] = outs[j]->data<double>();
        } else if (outs_type[j] == "float") {
          ptrs[num_ins + j] = outs[j]->data<float>();
        }
        args.push_back(&ptrs[num_ins + j]);
      }
      dev_code->Launch(n, &args);
    }
  }
};

}  // namespace operators
}  // namespace paddle
