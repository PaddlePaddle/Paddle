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

#include "paddle/phi/kernels/selected_rows/load_kernel.h"

#include <fstream>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/serialization.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void LoadKernel(const Context& dev_ctx,
                const std::string& file_path,
                int64_t seek,
                const std::vector<int64_t>& shape,
                bool load_as_fp16,
                SelectedRows* out) {
  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
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

  DeserializeFromStream(fin, out, dev_ctx);
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(load_sr, CPU, ALL_LAYOUT, phi::sr::LoadKernel, float) {}
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(load_sr, GPU, ALL_LAYOUT, phi::sr::LoadKernel, float) {}
#endif
