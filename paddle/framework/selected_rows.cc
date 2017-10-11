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

#include "paddle/framework/selected_rows.h"
#include "paddle/memory/memcpy.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace framework {

template <typename T>
void SelectedRowsToTensor(const SelectedRows& input,
                          const platform::Place& dst_place,
                          platform::DeviceContext& ctx, Tensor* output) {
  std::vector<int64_t> input_dims = vectorize(input.value().dims());
  input_dims.insert(input_dims.begin(), input.height());
  output->mutable_data<T>(dst_place, input_dims);
  auto src_place = input.place();
  auto rows = input.rows();

  auto row_numel = input.value().numel() / input.height();

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    operators::math::SetConstant<platform::CPUPlace, T>(ctx, output,
                                                        static_cast<T>(0.0));
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);

    for (size_t i = 0; i < rows.size(); i++) {
      memory::Copy(dst_cpu_place, output->data<T>() + rows[i] * row_numel,
                   src_cpu_place, input.value().data<T>() + i * row_numel,
                   row_numel * sizeof(T));
    }
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&
           platform::is_cpu_place(dst_place)) {
    PADDLE_THROW("Not supported");
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    PADDLE_THROW("Not supported");
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    PADDLE_THROW("Not supported");
  }

#endif
}

}  // namespace framework
}  // namespace paddle