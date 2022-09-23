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

template <typename DeviceContext>
static void MutableMultiTypeData(
    std::vector<paddle::framework::LoDTensor*>* var,
    const std::vector<int>& data_type,
    const DeviceContext& dev_ctx,
    const platform::Place& place) {
  for (size_t i = 0; i < var->size(); i++) {
    if (data_type[i] == framework::proto::VarType::FP32) {
      dev_ctx.template Alloc<float>((*var)[i],
                                    (*var)[i]->numel() * sizeof(float));
    } else if (data_type[i] == framework::proto::VarType::FP16) {
      dev_ctx.template Alloc<paddle::platform::float16>(
          (*var)[i], (*var)[i]->numel() * sizeof(paddle::platform::float16));
    } else if (data_type[i] == framework::proto::VarType::FP64) {
      dev_ctx.template Alloc<double>((*var)[i],
                                     (*var)[i]->numel() * sizeof(double));
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
    const auto& outs_dtype = ctx.Attr<std::vector<int>>("outs_dtype");
    const auto& inputs_dtype = ctx.Attr<std::vector<int>>("inputs_dtype");

    size_t num_ins = ins.size();
    size_t num_outs = outs.size();

    auto place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    MutableMultiTypeData(&outs, outs_dtype, dev_ctx, place);

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
        if (inputs_dtype[i] == framework::proto::VarType::FP16) {
          ptrs[i] = ins[i]->data<paddle::platform::float16>();
        } else if (inputs_dtype[i] == framework::proto::VarType::FP32) {
          ptrs[i] = ins[i]->data<float>();
        } else if (inputs_dtype[i] == framework::proto::VarType::FP64) {
          ptrs[i] = ins[i]->data<double>();
        }
        args.push_back(&ptrs[i]);
      }
      for (size_t j = 0; j < num_outs; ++j) {
        if (outs_dtype[j] == framework::proto::VarType::FP16) {
          ptrs[num_ins + j] = outs[j]->data<paddle::platform::float16>();
        } else if (outs_dtype[j] == framework::proto::VarType::FP32) {
          ptrs[num_ins + j] = outs[j]->data<float>();
        } else if (outs_dtype[j] == framework::proto::VarType::FP64) {
          ptrs[num_ins + j] = outs[j]->data<double>();
        }
        args.push_back(&ptrs[num_ins + j]);
      }
      dev_code->Launch(n, &args);
    }
  }
};

}  // namespace operators
}  // namespace paddle
