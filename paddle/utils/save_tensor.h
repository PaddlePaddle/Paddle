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

#pragma once

#include <cstdint>
#include <iostream>

#include <fstream>
#include <numeric>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/version.h"

namespace paddle {

void SaveTensor(const phi::DenseTensor& x,
                const std::string& file_path,
                bool overwrite,
                bool save_as_fp16) {
  std::string new_path(file_path);
  std::cout << "new path : " << new_path << std::endl;
  if (FileExists(new_path)) {
    std::cout << "FileExists : " << new_path << "pass" << std::endl;
    return;
  }

  VLOG(6) << "saved to " << new_path;
  MkDirRecursively(DirName(new_path).c_str());

  std::ofstream fout(new_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      phi::errors::Unavailable("Cannot open %s to save variables.", new_path));
  VLOG(6) << "START SerializeToStream";
  framework::SerializeToStream(fout, x);
  VLOG(6) << "end SerializeToStream";

  fout.close();
}

}  // namespace paddle
