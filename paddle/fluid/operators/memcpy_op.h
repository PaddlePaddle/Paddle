/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_context.h"

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {
class Variable;
class SelectedRows;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class MemcpyFunctor {
 private:
  enum DeviceType {
    CPU = 0,
    CUDA = 1,
    CUDA_PINNED = 2,
    XPU = 3,
    NPU = 4,
    NPU_PINNED = 5,
  };

 public:
  MemcpyFunctor(framework::Variable *out,
                const platform::DeviceContext &dev_ctx,
                const int dst_place_type)
      : out_(out), dev_ctx_(dev_ctx), dst_place_type_(dst_place_type) {}

  void operator()(const framework::LoDTensor &lod_tensor) const {
    auto &out_tensor = *out_->GetMutable<framework::LoDTensor>();

    if (dst_place_type_ == DeviceType::CUDA_PINNED) {
      framework::TensorCopy(lod_tensor, platform::CUDAPinnedPlace(), dev_ctx_,
                            &out_tensor);
    } else if (dst_place_type_ == DeviceType::CUDA) {
      framework::TensorCopy(lod_tensor, dev_ctx_.GetPlace(), dev_ctx_,
                            &out_tensor);
    } else if (dst_place_type_ == DeviceType::CPU) {
      framework::TensorCopySync(lod_tensor, platform::CPUPlace(), &out_tensor);
#ifdef PADDLE_WITH_ASCEND_CL
    } else if (dst_place_type_ == DeviceType::NPU) { /* npu_pin->npu */
      framework::TensorCopy(lod_tensor, dev_ctx_.GetPlace(), dev_ctx_,
                            &out_tensor);
    } else if (dst_place_type_ == DeviceType::NPU_PINNED) { /* npu->npu_pin */
      framework::TensorCopy(lod_tensor, platform::NPUPinnedPlace(), dev_ctx_,
                            &out_tensor);
#endif
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "memcpy dst_place_type: %d is not supported yet.", dst_place_type_));
    }
    out_tensor.set_lod(lod_tensor.lod());
  }

  void operator()(const pten::SelectedRows &rows) const {
    // (JZ-LIANG) to support SelectedRows
    PADDLE_THROW(platform::errors::Unimplemented(
        "Memcpy for SelectedRows is NOT support yet."));
  }

  template <typename T>
  void operator()(const T &v) const {
    PADDLE_ENFORCE_EQ(
        true, false,
        platform::errors::PermissionDenied(
            "Not support type for Memcpy  op with type %s", typeid(T).name()));
  }

 private:
  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
  const int dst_place_type_;
};

}  // namespace operators
}  // namespace paddle
