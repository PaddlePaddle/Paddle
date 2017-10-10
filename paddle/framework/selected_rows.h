/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

class SelectedRows {
 public:
  SelectedRows(const std::vector<int>& rows, const int height)
      : rows_(rows), height_(height) {}

  platform::Place place() { return value_.place(); }

  template <typename T>
  Tensor ToTensor() {
    Tensor dst;
    std::vector<int64_t> dims = vectorize(value_.dims());
    dims.insert(dims.beign(), height_);
    dst.Resize(make_ddim(dims));
    platform::Place place = place();
    dst.mutable_data<T>(place);
    auto* dst_ptr = static_cast<void*>(dst.mutable_data<T>(place));

    if (platform::is_cpu_place(place)) {
      platform::CPUPlace cpu_place = boost::get<platform::CPUPlace>(place);
      memset(dst_ptr, 0, dst.);
      memory::Copy(cpu_place, dst_ptr,
                   boost::get<platform::CPUPlace>(src_place), src_ptr, size);
    } else
      (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
// TODO(qijun): support GPU
#else
        PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
      }
  }

 private:
  std::vector<int> rows_;
  Tensor value_;
  int64_t height_;
};
}
}
