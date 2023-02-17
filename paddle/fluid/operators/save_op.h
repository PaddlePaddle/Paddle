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
#include "paddle/phi/kernels/cast_kernel.h"

namespace paddle {
namespace operators {

template <typename T, typename Context>
void SaveKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::string& file_path,
                bool overwrite,
                bool save_as_fp16) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      phi::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  MkDirRecursively(DirName(file_path).c_str());

  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      phi::errors::Unavailable("Cannot open %s to save variables.", file_path));

  auto in_dtype = x.dtype();
  auto out_dtype = save_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;

  if (in_dtype != out_dtype) {
    auto out = phi::Cast<T>(dev_ctx, x, out_dtype);
    framework::SerializeToStream(fout, out, dev_ctx);
  } else {
    framework::SerializeToStream(fout, x, dev_ctx);
  }
  fout.close();
}

template <typename T, typename Context>
void SaveSelectedRowsKernel(const Context& dev_ctx,
                            const phi::SelectedRows& x,
                            const std::string& file_path,
                            bool overwrite,
                            bool save_as_fp16) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      phi::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));
  PADDLE_ENFORCE_EQ(save_as_fp16,
                    false,
                    phi::errors::Unimplemented(
                        "SelectedRows is not supported to save as float16."));

  MkDirRecursively(DirName(file_path).c_str());

  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      phi::errors::Unavailable("Cannot open %s to save variables.", file_path));
  framework::SerializeToStream(fout, x, dev_ctx);
  fout.close();
}

// define LOOKUP_TABLE_PATH for checkpoint notify to save lookup table variables
// to directory specified.
constexpr char LOOKUP_TABLE_PATH[] = "kLookupTablePath";
template <typename DeviceContext, typename T>
class SaveOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();

    auto* input_var = ctx.InputVar("X");
    auto iname = ctx.InputNames("X").data();
    PADDLE_ENFORCE_NOT_NULL(
        input_var,
        phi::errors::InvalidArgument(
            "The variable %s to be saved cannot be found.", iname));

    auto filename = ctx.Attr<std::string>("file_path");
    auto overwrite = ctx.Attr<bool>("overwrite");
    auto save_as_fp16 = ctx.Attr<bool>("save_as_fp16");

    VLOG(4) << "save output file_path: " << filename;

    // get device context from pool
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(place);

    if (input_var->IsType<phi::DenseTensor>()) {
      auto& tensor = input_var->Get<phi::DenseTensor>();
      SaveKernel<T>(dev_ctx, tensor, filename, save_as_fp16);
    } else if (input_var->IsType<phi::SelectedRows>()) {
      auto& selectedRows = input_var->Get<phi::SelectedRows>();
      SaveSelectedRowsKernel<T>(dev_ctx, selectedRows, filename);
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Save operator only supports saving phi::DenseTensor and "
          "SelectedRows "
          "variable, %s has wrong type",
          iname));
    }
  }
};

}  // namespace operators
}  // namespace paddle
