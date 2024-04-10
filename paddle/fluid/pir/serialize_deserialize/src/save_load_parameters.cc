/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pir/serialize_deserialize/include/save_load_parameters.h"

#include <cstdint>
#include <fstream>
#include <numeric>

#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/common/port.h"

namespace pir {

void SaveFunction(const phi::DenseTensor& x,
                  const std::string& name,
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
  VLOG(6) << "save func save path: " << file_path;
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      phi::errors::Unavailable("Cannot open %s to save variables.", file_path));

  paddle::framework::SerializeToStream(fout, x);
  // TODO(changeyoung98): fp16
  fout.close();
  VLOG(6) << "save func done ";
}

void SaveCombineFunction(const std::vector<const phi::DenseTensor*>& x,
                         const std::vector<std::string>& names,
                         const std::string& file_path,
                         bool overwrite,
                         bool save_as_fp16,
                         bool save_to_memory) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      phi::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  MkDirRecursively(DirName(file_path).c_str());
  VLOG(6) << "save func save path: " << file_path;
  std::ostringstream ss;
  for (size_t i = 0; i < x.size(); i++) {
    auto& tensor = *(x[i]);
    PADDLE_ENFORCE_EQ(
        tensor.IsInitialized(),
        true,
        phi::errors::InvalidArgument(
            "The Tensor with Index (%d) to be saved is not initialized.", i));
    // TODO(changeyoung98): fp16
    paddle::framework::SerializeToStream(ss, tensor);
  }
  MkDirRecursively(DirName(file_path).c_str());
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      phi::errors::Unavailable("Cannot open %s to save variables.", file_path));
  fout << ss.str();
  fout.close();
  VLOG(6) << "save combine done ";
}

void LoadFunction(const std::string& file_path,
                  int64_t seek,
                  const std::vector<int64_t>& shape,
                  bool load_as_fp16,
                  phi::DenseTensor* out) {
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    phi::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "The variable to be loaded cannot be found."));

  if (seek != -1) {
    PADDLE_ENFORCE_GE(seek,
                      0,
                      phi::errors::InvalidArgument(
                          "seek with tensor must great than or equal to 0"));
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    const paddle::platform::DeviceContext* dev_ctx = nullptr;
    dev_ctx = pool.Get(paddle::platform::CPUPlace());
    paddle::framework::DeserializeFromStream(fin, out, *dev_ctx, seek, shape);
  } else {
    paddle::framework::DeserializeFromStream(fin, out);
  }

  // TODO(changeyoung98): fp16
}

void LoadCombineFunction(const std::string& file_path,
                         const std::vector<std::string>& names,
                         std::vector<phi::DenseTensor*>* out,
                         bool load_as_fp16) {
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    phi::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  for (size_t i = 0; i < names.size(); i++) {
    auto tensor = out->at(i);
    paddle::framework::DeserializeFromStream(fin, tensor);
  }
  fin.peek();
  PADDLE_ENFORCE_EQ(
      fin.eof(),
      true,
      phi::errors::Unavailable("Not allowed to load partial data via "
                               "load_combine_op, please use load_op instead."));
  // TODO(changeyoung98): fp16
}

}  // namespace pir
