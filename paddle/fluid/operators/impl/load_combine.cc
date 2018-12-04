// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/impl/load_combine.h"

namespace paddle {
namespace operators {
namespace impl {

void LoadParamsFromStream(const std::vector<std::string> &out_var_names,
                          const paddle::platform::Place &place,
                          bool load_as_fp16, std::istream *buffer,
                          const paddle::framework::Scope *scope) {
  auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  for (size_t i = 0; i < out_var_names.size(); i++) {
    auto *out_var = scope->FindVar(out_var_names[i]);

    PADDLE_ENFORCE(out_var != nullptr, "Output variable %s cannot be found",
                   out_var_names[i]);

    auto *tensor = out_var->GetMutable<framework::LoDTensor>();

    // Get data from fin to tensor
    DeserializeFromStream(*buffer, tensor, *dev_ctx);

    auto in_dtype = framework::ToDataType(tensor->type());
    auto out_dtype = load_as_fp16 ? framework::proto::VarType::FP16 : in_dtype;

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
      out_var->Clear();
      tensor = out_var->GetMutable<framework::LoDTensor>();
      tensor->set_lod(fp16_tensor.lod());
      tensor->ShareDataWith(fp16_tensor);
    }
  }
}

}  // namespace impl
}  // namespace operators
}  // namespace paddle
