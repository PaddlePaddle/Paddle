//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <utility>

#include "paddle/tcmpt/core/tensor_interface.h"
#include "paddle/utils/any.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace pt {

using DeviceContext = paddle::platform::DeviceContext;

/**
 * Note: KernelContext doesn't manage the life if DeviceContext and Tensor
 *
 * Note: KernelContext does not couple the concept of framework,
 *       its constructor can only take the members it needs as parameters,
 *       not Scope, RuntimeContext, etc. as parameters
 */
class KernelContext {
 public:
  explicit KernelContext(const DeviceContext& dev_ctx) : dev_ctx_(dev_ctx) {}
  KernelContext(const DeviceContext& dev_ctx,
                const std::vector<std::shared_ptr<TensorInterface>>& inputs,
                const std::vector<std::shared_ptr<TensorInterface>>& outputs,
                const std::vector<paddle::any>& attrs)
      : dev_ctx_(dev_ctx), inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  template <typename CtxType>
  const CtxType& GetDeviceContext() const {
    return static_cast<const CtxType&>(dev_ctx_);
  }

  void EmplaceBackInput(std::shared_ptr<TensorInterface> input) {
    inputs_.emplace_back(input);
  }

  void EmplaceBackOutput(std::shared_ptr<TensorInterface> output) {
    outputs_.emplace_back(output);
  }

  void EmplaceBackAttr(paddle::any attr) { attrs_.emplace_back(attr); }

  template <typename TensorType>
  const TensorType& InputAt(size_t idx) const {
    return static_cast<const TensorType&>(*(inputs_.at(idx)));
  }

  template <typename TensorType>
  TensorType* MutableOutputAt(size_t idx) {
    return static_cast<TensorType*>(outputs_.at(idx).get());
  }

  template <typename AttrType>
  AttrType AttrAt(size_t idx) const {
    try {
      return paddle::any_cast<AttrType>(attrs_.at(idx));
    } catch (paddle::bad_any_cast&) {
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Attribute cast error in Op Kernel Context."));
    }
  }

 private:
  // DeviceContext base class
  const DeviceContext& dev_ctx_;

  // TODO(chenweihang): replaced by small_vector
  // TODO(chenweihang): Tensor -> Tensor*, Tensor should by managed `scope`
  // Note: can't use API Tensor here, the inference don't use this API Tensor
  std::vector<std::shared_ptr<TensorInterface>> inputs_{};
  std::vector<std::shared_ptr<TensorInterface>> outputs_{};
  std::vector<paddle::any> attrs_{};

  // Only contains input like list[Tensor] need `range`
  // TODO(chenweihang): replaced by small_vector
  std::vector<std::pair<int, int>> input_range_{{}};
  std::vector<std::pair<int, int>> output_range_{{}};

  // Only static graph need `name`
  // TODO(chenweihang): replaced by paddle::string_view
  std::vector<std::string> input_names_{{}};
  std::vector<std::string> output_names_{{}};
};

}  // namespace pt
