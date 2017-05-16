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
#include "AttributeMeta.h"
#include "TensorMeta.h"

namespace paddle {
namespace topology {
namespace meta {

template <typename T>
static bool checkHelper(const AttributeMeta* meta,
                        any* attr,
                        bool alreadySet,
                        Error* err) {
  if (attr->type() == typeid(T)) {
    *err = meta->validate(any_cast<T>(attr), alreadySet);
    return true;
  } else if (meta->type == typeid(T) && attr->type() == typeid(void) &&
             !alreadySet) {
    *attr = T();
    *err = meta->validate(any_cast<T>(attr), alreadySet);
    return true;
  } else {
    return false;
  }
}

#define CHECK_HELPER(TYPE)                                 \
  do {                                                     \
    paddle::Error err;                                     \
    if (checkHelper<TYPE>(this, attr, alreadySet, &err)) { \
      return err;                                          \
    }                                                      \
  } while (0)

Error AttributeMeta::validate(any* attr, bool alreadySet) const {
  if (attr->type() != this->type && attr->type() != typeid(void)) {
    return Error("Type mismatch, expect %s, actual %s",
                 this->type.name(),
                 attr->type().name());
  }
  /**
   * Only following types are supported as an attribute.
   */
  CHECK_HELPER(int);
  CHECK_HELPER(double);
  CHECK_HELPER(std::vector<size_t>);
  CHECK_HELPER(bool);
  return paddle::Error("Unsupported attribute type %s", attr->type().name());
}

}  // namespace meta
}  // namespace topology
}  // namespace paddle
