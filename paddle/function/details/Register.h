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

#pragma once
#include <functional>
#include "../FunctionList.h"
#include "../TensorType.h"
#include "paddle/topology/Attribute.h"
#include "paddle/topology/Function.h"
#include "paddle/topology/meta/FunctionMeta.h"
#include "paddle/utils/Util.h"

namespace paddle {
namespace function {
namespace details {
class FunctionRegister {
public:
  FunctionRegister(topology::meta::FunctionMetaPtr& meta) : meta_(meta) {}

  template <typename T, DeviceType devType>
  paddle::Error reg(std::function<Error(const BufferArgs& ins,
                                        const BufferArgs& outs,
                                        const T& attrs)> kernel) {
    auto meta = meta_;
    auto key = devType == DEVICE_TYPE_CPU ? "CPUKernel" : "GPUKernel";
    auto inited = std::make_shared<bool>(false);
    auto tmp = std::make_shared<T>();
    FunctionWithAttrs fn = [kernel, meta, inited, tmp](
        const BufferArgs& ins,
        const BufferArgs& outs,
        const topology::AttributeMap& attrs) {
      bool& init = *inited;
      if (!init) {
        auto err =
            meta->parseAttribute<paddle::topology::Attribute>(attrs, tmp.get());
        if (!err.isOK()) return err;
        init = true;
      }
      return kernel(ins, outs, *tmp);
    };
    return meta_->metaAttributes_.set(key, fn);
  }

private:
  topology::meta::FunctionMetaPtr& meta_;
};

}  // namespace details

}  // namespace function
}  // namespace paddle
