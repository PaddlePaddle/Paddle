/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/framework/type_deducer.h"

namespace paddle {
namespace framework {

template <>
Tensor* TypeDeducer::operator()<Tensor>(const std::string& name) const {
  const auto& tensor_type = typeid(Tensor);
  auto it = rules.find(tensor_type);
  PADDLE_ENFORCE(it != rules.end(), "no type deduce rules for target type [%s]",
                 typeid(Tensor).name());
  for (auto& rule : it->second) {
    rule->Init(this);
    if (rule->Match()) {
      return (*rule)(name, tensor_type)->GetMutable<Tensor>();
    }
  }
  return nullptr;
}

TypeDeducer::TypeDeducer() {
  auto tensor2tensor =
      std::unique_ptr<TypeDeduceRule>(new Tensor2TensorDeduceRule);
  auto lottensor2tensor =
      std::unique_ptr<LOT2TensorDeduceRule>(new LOT2TensorDeduceRule);
  AddRule(typeid(Tensor), std::move(tensor2tensor));
  AddRule(typeid(Tensor), std::move(lottensor2tensor));
}

}  // namespace framework
}  // namespace paddle
