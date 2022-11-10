// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/kernels/save_combine_kernel.h"

#include <stdint.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/string_array.h"

#include "paddle/phi/backends/dynload/port.h"
#include "paddle/phi/core/serialization.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"

namespace phi {

inline void SaveToMemory(const std::string& file_path,
                         const std::ostringstream& ss,
                         bool save_to_memory,
                         std::string* output) {
  if (save_to_memory) {
    PADDLE_ENFORCE_NE(output,
                      nullptr,
                      phi::errors::InvalidArgument(
                          "Cannot find variable Y for save_combine_op"));
    *output = ss.str();
  } else {
    MkDirRecursively(DirName(file_path).c_str());
    std::ofstream fout(file_path, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                      true,
                      phi::errors::Unavailable(
                          "Cannot open %s to save variables.", file_path));
    fout << ss.str();
    fout.close();
  }
}

template <typename T, typename Context>
void SaveCombineTensorKernel(const Context& dev_ctx,
                             const std::vector<const DenseTensor*>& x,
                             const std::string& file_path,
                             bool overwrite,
                             bool save_as_fp16,
                             bool save_to_memory,
                             std::string* y) {
  bool is_present = FileExists(file_path);
  if (is_present && !overwrite) {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "%s exists! Cannot save_combine to it when overwrite is set to "
        "false.",
        file_path,
        overwrite));
  }

  std::ostringstream ss;
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The number of variables to be saved is %d, expect "
                        "it to be greater than 0.",
                        x.size()));

  for (size_t i = 0; i < x.size(); i++) {
    auto& tensor = *(x[i]);
    PADDLE_ENFORCE_EQ(
        tensor.IsInitialized(),
        true,
        phi::errors::InvalidArgument(
            "The Tensor with Index (%d) to be saved is not initialized.", i));
    // Serialize tensors one by one
    // Check types to see if a fp16 transformation is required
    auto in_dtype = tensor.dtype();
    auto out_dtype = save_as_fp16 ? DataType::FLOAT16 : in_dtype;

    if (in_dtype != out_dtype) {
      phi::DenseTensor out =
          phi::funcs::TransDataType(dev_ctx, tensor, out_dtype);
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
void SaveCombineVocabKernel(const Context& dev_ctx,
                            const std::vector<const Vocab*>& x,
                            const std::string& file_path,
                            bool overwrite,
                            bool save_as_fp16,
                            bool save_to_memory,
                            std::string* y) {
  bool is_present = FileExists(file_path);
  if (is_present && !overwrite) {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "%s exists! Cannot save_combine to it when overwrite is set to "
        "false.",
        file_path,
        overwrite));
  }

  std::ostringstream ss;
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    phi::errors::InvalidArgument(
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

}  // namespace phi
