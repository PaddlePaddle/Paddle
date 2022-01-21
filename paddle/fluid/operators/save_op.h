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

#include <stdint.h>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace operators {
// define LOOKUP_TABLE_PATH for checkpoint notify to save lookup table variables
// to directory specified.
constexpr char LOOKUP_TABLE_PATH[] = "kLookupTablePath";
template <typename DeviceContext, typename T>
class SaveOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();

    auto *input_var = ctx.InputVar("X");
    auto iname = ctx.InputNames("X").data();
    PADDLE_ENFORCE_NOT_NULL(
        input_var, platform::errors::InvalidArgument(
                       "The variable %s to be saved cannot be found.", iname));

    auto filename = ctx.Attr<std::string>("file_path");
    auto overwrite = ctx.Attr<bool>("overwrite");

    VLOG(4) << "save output file_path: " << filename;

    PADDLE_ENFORCE_EQ(
        FileExists(filename) && !overwrite, false,
        platform::errors::PreconditionNotMet(
            "%s exists!, cannot save to it when overwrite is set to false.",
            filename, overwrite));

    MkDirRecursively(DirName(filename).c_str());

    if (input_var->IsType<framework::LoDTensor>()) {
      SaveLodTensor(ctx, place, input_var, filename);
    } else if (input_var->IsType<framework::SelectedRows>()) {
      SaveSelectedRows(ctx, place, input_var, filename);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Save operator only supports saving LoDTensor and SelectedRows "
          "variable, %s has wrong type",
          iname));
    }
  }

  void SaveLodTensor(const framework::ExecutionContext &ctx,
                     const platform::Place &place,
                     const framework::Variable *var,
                     const std::string &filename) const {
    auto &tensor = var->Get<framework::LoDTensor>();

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    std::ofstream fout(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                      platform::errors::Unavailable(
                          "Cannot open %s to save variables.", filename));

    auto save_as_fp16 = ctx.Attr<bool>("save_as_fp16");
    auto in_dtype = tensor.type();
    auto out_dtype = save_as_fp16 ? framework::proto::VarType::FP16 : in_dtype;

    if (in_dtype != out_dtype) {
      auto in_kernel_type = framework::OpKernelType(in_dtype, place);
      auto out_kernel_type = framework::OpKernelType(out_dtype, place);
      framework::LoDTensor out;
      framework::TransDataType(in_kernel_type, out_kernel_type, tensor, &out);
      // copy LoD info to the new tensor
      out.set_lod(tensor.lod());
      framework::SerializeToStream(fout, out, dev_ctx);
    } else {
      framework::SerializeToStream(fout, tensor, dev_ctx);
    }
    fout.close();
  }

  void SaveSelectedRows(const framework::ExecutionContext &ctx,
                        const platform::Place &place,
                        const framework::Variable *var,
                        const std::string &filename) const {
    auto &selectedRows = var->Get<framework::SelectedRows>();

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    // FIXME(yuyang18): We save variable to local file now, but we should change
    // it to save an output stream.
    std::ofstream fout(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                      platform::errors::Unavailable(
                          "Cannot open %s to save variables.", filename));
    framework::SerializeToStream(fout, selectedRows, dev_ctx);
    fout.close();
  }
};

}  // namespace operators
}  // namespace paddle
