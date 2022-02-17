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

#include <stdint.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/backends/dynload/port.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class SaveCombineOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    auto filename = ctx.Attr<std::string>("file_path");
    auto overwrite = ctx.Attr<bool>("overwrite");
    auto save_as_fp16 = ctx.Attr<bool>("save_as_fp16");
    auto save_to_memory = ctx.Attr<bool>("save_to_memory");
    auto output = ctx.Output<std::string>("Y");

    bool is_present = FileExists(filename);
    if (is_present && !overwrite) {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "%s exists! Cannot save_combine to it when overwrite is set to "
          "false.",
          filename, overwrite));
    }

    std::ostringstream ss;
    auto inp_var_names = ctx.InputNames("X");
    auto &inp_vars = ctx.MultiInputVar("X");
    PADDLE_ENFORCE_GT(inp_var_names.size(), 0UL,
                      platform::errors::InvalidArgument(
                          "The number of variables to be saved is %d, expect "
                          "it to be greater than 0.",
                          inp_var_names.size()));

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    for (size_t i = 0; i < inp_var_names.size(); i++) {
      PADDLE_ENFORCE_NOT_NULL(
          inp_vars[i],
          platform::errors::InvalidArgument("Cannot find variable %s to save.",
                                            inp_var_names[i]));
      PADDLE_ENFORCE_EQ(inp_vars[i]->IsType<framework::LoDTensor>() ||
                            inp_vars[i]->IsType<framework::Vocab>(),
                        true,
                        platform::errors::InvalidArgument(
                            "SaveCombine operator only supports saving "
                            "LoDTensor or Vocab variable, %s has wrong type.",
                            inp_var_names[i]));

      if (inp_vars[i]->IsType<framework::LoDTensor>()) {
        auto &tensor = inp_vars[i]->Get<framework::LoDTensor>();
        PADDLE_ENFORCE_EQ(
            tensor.IsInitialized(), true,
            platform::errors::InvalidArgument(
                "The Tensor of Variable(%s) to be saved is not initialized.",
                inp_var_names[i]));
        // Serialize tensors one by one
        // Check types to see if a fp16 transformation is required
        auto in_dtype = framework::TransToProtoVarType(tensor.dtype());
        auto out_dtype =
            save_as_fp16 ? framework::proto::VarType::FP16 : in_dtype;

        if (in_dtype != out_dtype) {
          auto in_kernel_type = framework::OpKernelType(in_dtype, place);
          auto out_kernel_type = framework::OpKernelType(out_dtype, place);
          framework::LoDTensor out;
          // copy LoD info to the new tensor
          out.set_lod(tensor.lod());
          framework::TransDataType(in_kernel_type, out_kernel_type, tensor,
                                   &out);
          framework::SerializeToStream(ss, out, dev_ctx);
        } else {
          framework::SerializeToStream(ss, tensor, dev_ctx);
        }
      } else {
        auto &tensor = inp_vars[i]->Get<framework::Vocab>();
        std::unordered_map<std::string, std::int32_t> data;
        for (auto it = tensor.begin(); it != tensor.end(); ++it) {
          std::string t;
          framework::ConvertWstrToStr(it->first, &t);
          data.emplace(t, it->second);
        }
        framework::StringMapToStream(ss, data);
      }
    }
    if (save_to_memory) {
      PADDLE_ENFORCE_NE(output, nullptr,
                        platform::errors::InvalidArgument(
                            "Cannot find variable Y for save_combine_op"));
      *output = ss.str();
    } else {
      MkDirRecursively(DirName(filename).c_str());
      std::ofstream fout(filename, std::ios::binary);
      PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                        platform::errors::Unavailable(
                            "Cannot open %s to save variables.", filename));
      fout << ss.str();
      fout.close();
    }
  }
};

}  // namespace operators
}  // namespace paddle
