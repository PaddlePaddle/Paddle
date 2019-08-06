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

#include <fstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class LoadCombineOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto filename = ctx.Attr<std::string>("file_path");
    auto load_as_fp16 = ctx.Attr<bool>("load_as_fp16");
    auto model_from_memory = ctx.Attr<bool>("model_from_memory");
    auto &out_var_names = ctx.Outputs("Out");

    PADDLE_ENFORCE_GT(
        static_cast<int>(out_var_names.size()), 0,
        "The number of output variables should be greater than 0.");
    if (!model_from_memory) {
      std::ifstream fin(filename, std::ios::binary);
      PADDLE_ENFORCE(static_cast<bool>(fin),
                     "Cannot open file %s for load_combine op", filename);
      LoadParamsFromBuffer(ctx, place, &fin, load_as_fp16, out_var_names);
    } else {
      PADDLE_ENFORCE(!filename.empty(), "Cannot load file from memory");
      std::stringstream fin(filename, std::ios::in | std::ios::binary);
      LoadParamsFromBuffer(ctx, place, &fin, load_as_fp16, out_var_names);
    }
  }

  void LoadParamsFromBuffer(
      const framework::ExecutionContext &context, const platform::Place &place,
      std::istream *buffer, bool load_as_fp16,
      const std::vector<std::string> &out_var_names) const {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);
    auto out_vars = context.MultiOutputVar("Out");

    for (size_t i = 0; i < out_var_names.size(); i++) {
      PADDLE_ENFORCE(out_vars[i] != nullptr,
                     "Output variable %s cannot be found", out_var_names[i]);

      auto *tensor = out_vars[i]->GetMutable<framework::LoDTensor>();

      // Error checking
      PADDLE_ENFORCE(
          static_cast<bool>(*buffer),
          "There is a problem with loading model parameters. "
          "Please check whether the model file is complete or damaged.");

      // Get data from fin to tensor
      DeserializeFromStream(*buffer, tensor, dev_ctx);

      auto in_dtype = tensor->type();
      auto out_dtype =
          load_as_fp16 ? framework::proto::VarType::FP16 : in_dtype;

      if (in_dtype != out_dtype) {
        // convert to float16 tensor
        auto in_kernel_type = framework::OpKernelType(in_dtype, place);
        auto out_kernel_type = framework::OpKernelType(out_dtype, place);
        framework::LoDTensor fp16_tensor;
        // copy LoD info to the new tensor
        fp16_tensor.set_lod(tensor->lod());
        framework::TransDataType(in_kernel_type, out_kernel_type, *tensor,
                                 &fp16_tensor);

        // reset output tensor
        out_vars[i]->Clear();
        tensor = out_vars[i]->GetMutable<framework::LoDTensor>();
        tensor->set_lod(fp16_tensor.lod());
        tensor->ShareDataWith(fp16_tensor);
      }
    }
    buffer->peek();
    PADDLE_ENFORCE(buffer->eof(),
                   "You are not allowed to load partial data via "
                   "load_combine_op, use load_op instead.");
  }
};

}  // namespace operators
}  // namespace paddle
