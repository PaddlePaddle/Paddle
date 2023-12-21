// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <string>
#include <vector>

#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ReadFileKernel(const Context& dev_ctx,
                    const std::string& filename,
                    DenseTensor* out) {
  std::ifstream input(filename.c_str(),
                      std::ios::in | std::ios::binary | std::ios::ate);
  std::streamsize file_size = input.tellg();

  input.seekg(0, std::ios::beg);

  out->Resize({file_size});
  uint8_t* data = dev_ctx.template Alloc<T>(out);
  input.read(reinterpret_cast<char*>(data), file_size);
}
}  // namespace phi

PD_REGISTER_KERNEL(read_file, CPU, ALL_LAYOUT, phi::ReadFileKernel, uint8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UINT8);
}
