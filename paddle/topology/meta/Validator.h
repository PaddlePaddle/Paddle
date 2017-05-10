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
#include <unordered_map>
#include "../Function.h"
#include "../Tensor.h"
#include "AttributeMeta.h"
#include "paddle/utils/Any.h"
namespace paddle {
namespace topology {
namespace meta {

class AttributeValidator {
public:
  AttributeValidator(
      const std::unordered_map<std::string, AttributeMetaPtr>& attributeMetas)
      : metas_(attributeMetas) {}

  paddle::Error validate(std::unordered_map<std::string, any>* attrs) const;

private:
  const std::unordered_map<std::string, AttributeMetaPtr>& metas_;
};

paddle::Error validateAndInferShape(paddle::topology::Function& func,
                                    bool validOutput = false);
paddle::Error validate(const WithAttributeMeta& meta,
                       paddle::topology::Attribute& attr);

}  // namespace meta
}  // namespace topology
}  // namespace paddle
