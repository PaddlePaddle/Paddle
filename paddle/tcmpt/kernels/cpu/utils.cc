/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/tcmpt/kernels/cpu/utils.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/core/dtype.h"

namespace pt {

void Copy(const CPUContext& dev_ctx, const DenseTensor& src, DenseTensor* dst) {
  auto* src_ptr = src.data();
  auto* dst_ptr = dst->mutable_data();
  const auto& src_place = src.place();
  const auto& dst_place = dst->place();
  src.CheckMemorySize();

  if (src_ptr == dst_ptr && src_place == dst_place) {
    VLOG(3) << "Skip copy the same data async from " << src_place << " to "
            << dst_place;
    return;
  }
  VLOG(4) << "src:" << src_ptr << ", dst:" << dst_ptr;

  VLOG(3) << "TensorCopy " << src.dims() << " from " << src.place() << " to "
          << dst_place;
  dst->Resize(src.dims());
  dst->mutable_meta()->layout = src.meta().layout;
  auto size = src.numel() *
              paddle::framework::SizeOfType(TransToProtoVarType(src.type()));

  if (paddle::platform::is_cpu_place(src_place) &&
      paddle::platform::is_cpu_place(dst_place)) {
    paddle::memory::Copy(BOOST_GET_CONST(paddle::platform::CPUPlace, dst_place),
                         dst_ptr,
                         BOOST_GET_CONST(paddle::platform::CPUPlace, src_place),
                         src_ptr,
                         size);
  }
}

}  // namespace pt

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(UtilsCPU);

PT_REGISTER_KERNEL_WITH_NO_TYPE("copy", CPU, Any, pt::Copy) {}
