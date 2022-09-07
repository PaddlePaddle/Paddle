/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/selected_rows/save_kernel.h"

#include <fstream>

#include "paddle/phi/backends/dynload/port.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/serialization.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void SaveKernel(const Context& dev_ctx,
                const SelectedRows& x,
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
  SerializeToStream(fout, x, dev_ctx);
  fout.close();
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(save_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::SaveKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(save_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::SaveKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
