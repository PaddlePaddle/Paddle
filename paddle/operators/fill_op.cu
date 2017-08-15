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

#include "paddle/memory/memcpy.h"
#include "paddle/operators/fill_op.h"
namespace paddle {
namespace operators {
template <typename T>
class FillOpGPUKernel : public FillOpKernelBase<T> {
 public:
  void Copy(const platform::Place &place, const std::vector<T> &src,
            T *dst) const override {
    auto &gpu_place = boost::get<platform::GPUPlace>(place);
    auto &cpu_place = platform::default_cpu();
    memory::Copy(gpu_place, dst, cpu_place, src.data(), src.size() * sizeof(T));
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_GPU_KERNEL(fill, paddle::operators::FillOpGPUKernel<float>);
