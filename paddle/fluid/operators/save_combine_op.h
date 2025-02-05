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
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/raw_tensor.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace paddle {
namespace operators {

inline void SaveToMemory(const std::string& file_path,
                         const std::ostringstream& ss,
                         bool save_to_memory,
                         std::string* output) {
  if (save_to_memory) {
    PADDLE_ENFORCE_NE(output,
                      nullptr,
                      common::errors::InvalidArgument(
                          "Cannot find variable Y for save_combine_op"));
    *output = ss.str();
  } else {
    MkDirRecursively(DirName(file_path).c_str());
    std::ofstream fout(file_path, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                      true,
                      common::errors::Unavailable(
                          "Cannot open %s to save variables.", file_path));
    fout << ss.str();
    fout.close();
  }
}

template <typename T, typename Context>
void SaveCombineTensorKernel(const Context& dev_ctx,
                             const std::vector<const phi::DenseTensor*>& x,
                             const std::string& file_path,
                             bool overwrite,
                             bool save_as_fp16,
                             bool save_to_memory,
                             phi::ExtendedTensor* out) {
  std::string* y = nullptr;
  if (out != nullptr) {
    auto raw_out = static_cast<paddle::framework::RawTensor*>(out);
    y = raw_out->GetMutable<std::string>();
  }

  bool is_present = FileExists(file_path);
  if (is_present && !overwrite) {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "%s exists! Cannot save_combine to it when overwrite is set to "
        "false.",
        file_path,
        overwrite));
  }

  std::ostringstream ss;
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The number of variables to be saved is %d, expect "
                        "it to be greater than 0.",
                        x.size()));

  for (size_t i = 0; i < x.size(); i++) {
    auto& tensor = *(x[i]);
    PADDLE_ENFORCE_EQ(
        tensor.IsInitialized(),
        true,
        common::errors::InvalidArgument(
            "The Tensor with Index (%d) to be saved is not initialized.", i));
    // Serialize tensors one by one
    // Check types to see if a fp16 transformation is required
    auto in_dtype = tensor.dtype();
    auto out_dtype = save_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
    if (in_dtype != out_dtype) {
      auto place = dev_ctx.GetPlace();
      auto in_kernel_type =
          phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, in_dtype);
      auto out_kernel_type =
          phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, out_dtype);
      phi::DenseTensor out;
      framework::TransDataType(in_kernel_type, out_kernel_type, tensor, &out);
      // copy LoD info to the new tensor
      out.set_lod(tensor.lod());
      phi::SerializeToStream(ss, out, dev_ctx);
    } else {
      phi::SerializeToStream(ss, tensor, dev_ctx);
    }
  }

  SaveToMemory(file_path, ss, save_to_memory, y);
}

template <typename T, typename Context>
void SaveCombineVocabKernel(
    const Context& dev_ctx UNUSED,
    const std::vector<const phi::ExtendedTensor*>& inputs,
    const std::string& file_path,
    bool overwrite,
    bool save_as_fp16 UNUSED,
    bool save_to_memory,
    phi::ExtendedTensor* out) {
  std::string* y = nullptr;
  if (out != nullptr) {
    auto raw_out = static_cast<paddle::framework::RawTensor*>(out);
    y = raw_out->GetMutable<std::string>();
  }

  std::vector<const framework::Vocab*> x;
  x.reserve(inputs.size());
  for (auto input : inputs) {
    x.push_back(static_cast<const framework::Vocab*>(input));
  }
  bool is_present = FileExists(file_path);
  if (is_present && !overwrite) {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "%s exists! Cannot save_combine to it when overwrite is set to "
        "false.",
        file_path,
        overwrite));
  }

  std::ostringstream ss;
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The number of variables to be saved is %d, expect "
                        "it to be greater than 0.",
                        x.size()));

  for (size_t i = 0; i < x.size(); i++) {
    auto& tensor = *(x[i]);
    std::unordered_map<std::string, std::int32_t> data;
    for (auto it = tensor.begin(); it != tensor.end(); ++it) {
      std::string t;
      paddle::framework::ConvertWstrToStr(it->first, &t);
      data.emplace(t, it->second);
    }
    paddle::framework::StringMapToStream(ss, data);
  }

  SaveToMemory(file_path, ss, save_to_memory, y);
}

template <typename DeviceContext, typename T>
class SaveCombineOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    auto filename = ctx.Attr<std::string>("file_path");
    auto overwrite = ctx.Attr<bool>("overwrite");
    auto save_as_fp16 = ctx.Attr<bool>("save_as_fp16");
    auto save_to_memory = ctx.Attr<bool>("save_to_memory");
    auto output = ctx.Output<framework::RawTensor>("Y");
    auto inp_var_names = ctx.InputNames("X");
    auto& inp_vars = ctx.MultiInputVar("X");

    PADDLE_ENFORCE_GT(inp_var_names.size(),
                      0UL,
                      common::errors::InvalidArgument(
                          "The number of variables to be saved is %d, expect "
                          "it to be greater than 0.",
                          inp_var_names.size()));

    // get device context from pool
    phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(place);

    if (inp_vars.size() > 0 && inp_vars[0]->IsType<phi::DenseTensor>()) {
      std::vector<const phi::DenseTensor*> x(inp_vars.size());
      for (size_t i = 0; i < inp_vars.size(); i++) {
        PADDLE_ENFORCE_NOT_NULL(
            inp_vars[i],
            common::errors::InvalidArgument("Cannot find variable %s to save.",
                                            inp_var_names[i]));
        PADDLE_ENFORCE_EQ(
            inp_vars[i]->IsType<phi::DenseTensor>(),
            true,
            common::errors::InvalidArgument(
                "SaveCombine operator only supports saving "
                "phi::DenseTensor or Vocab variable, %s has wrong type.",
                inp_var_names[i]));
        x[i] = (&(inp_vars[i]->Get<phi::DenseTensor>()));
      }
      SaveCombineTensorKernel<T>(dev_ctx,
                                 x,
                                 filename,
                                 overwrite,
                                 save_as_fp16,
                                 save_to_memory,
                                 output);
    } else {
      std::vector<const phi::ExtendedTensor*> x(inp_vars.size());
      for (size_t i = 0; i < inp_vars.size(); i++) {
        PADDLE_ENFORCE_NOT_NULL(
            inp_vars[i],
            common::errors::InvalidArgument("Cannot find variable %s to save.",
                                            inp_var_names[i]));
        PADDLE_ENFORCE_EQ(
            inp_vars[i]->IsType<framework::Vocab>(),
            true,
            common::errors::InvalidArgument(
                "SaveCombine operator only supports saving "
                "phi::DenseTensor or Vocab variable, %s has wrong type.",
                inp_var_names[i]));
        x[i] = (&(inp_vars[i]->Get<framework::Vocab>()));
      }
      SaveCombineVocabKernel<T>(dev_ctx,
                                x,
                                filename,
                                overwrite,
                                save_as_fp16,
                                save_to_memory,
                                output);
    }
  }
};

}  // namespace operators
}  // namespace paddle
