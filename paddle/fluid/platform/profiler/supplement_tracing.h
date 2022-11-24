/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

<<<<<<< HEAD
#include <string>
=======
#include <map>
#include <string>
#include <utility>
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/profiler/trace_event.h"
#include "paddle/phi/core/compat/arg_map_context.h"

namespace paddle {

namespace framework {
class RuntimeContext;
}
namespace platform {

class RecordOpInfoSupplement {
 public:
<<<<<<< HEAD
=======
  static bool IsEnabled();

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  /**
   * @param type:  Operator type name.
   * @param attrs: Attribute map of op.
   * @param shape_ctx: Infershape context object.
   * @param ctx: Runtime context object.
   */
  explicit RecordOpInfoSupplement(const std::string& type,
                                  const framework::AttributeMap& attrs,
                                  const framework::InferShapeContext& shape_ctx,
                                  const framework::RuntimeContext& ctx);
  /**
   * @param type:  Operator type name.
   * @param attrs: Attribute map of op.
   * @param shape_ctx: Infershape context object.
   * @param kernel_signature:  KernelSignature object, used in dygraph.
   */
  explicit RecordOpInfoSupplement(const std::string& type,
                                  const framework::AttributeMap& attrs,
                                  const framework::InferShapeContext& shape_ctx,
                                  const phi::KernelSignature& kernel_signature);
<<<<<<< HEAD
=======

  /**
   *
   */
  explicit RecordOpInfoSupplement(
      const std::string& type,
      const std::vector<std::pair<const char*, std::vector<framework::DDim>>>&
          input_shapes);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
};

}  // namespace platform
}  // namespace paddle
