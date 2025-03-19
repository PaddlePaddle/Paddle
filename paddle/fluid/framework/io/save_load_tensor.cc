/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/io/save_load_tensor.h"

#include <cstdint>
#include <fstream>
#include <numeric>

#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/common/port.h"

namespace paddle::framework {

void SaveTensor(const phi::DenseTensor& x,
                const std::string& file_path,
                bool overwrite) {
  std::string new_path(file_path);

  VLOG(6) << "tensor will be saved to " << new_path;
  MkDirRecursively(DirName(new_path).c_str());

  std::ofstream fout(new_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", new_path));
  phi::SerializeToStream(fout, x);

  fout.close();
}

void LoadTensor(const std::string& file_path, phi::DenseTensor* out) {
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    common::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  PADDLE_ENFORCE_NOT_NULL(out,
                          common::errors::InvalidArgument(
                              "The variable to be loaded cannot be found."));

  phi::DeserializeFromStream(fin, out);
}
}  // namespace paddle::framework
