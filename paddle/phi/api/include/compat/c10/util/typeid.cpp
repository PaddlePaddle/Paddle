// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include <c10/util/typeid.h>

#include <algorithm>

namespace caffe2 {

std::mutex& TypeMeta::getTypeMetaDatasLock() {
  static std::mutex lock;
  return lock;
}

uint16_t TypeMeta::nextTypeIndex(
    static_cast<uint16_t>(c10::ScalarType::NumOptions));

detail::TypeMetaData* TypeMeta::typeMetaDatas() {
  static detail::TypeMetaData instances[kMaxTypeIndex + 1] = {
#define SCALAR_TYPE_META(T, _2, name)                     \
  detail::TypeMetaData(sizeof(T),                         \
                       detail::_PickNew<T>(),             \
                       detail::_PickPlacementNew<T>(),    \
                       detail::_PickCopy<T>(),            \
                       detail::_PickPlacementDelete<T>(), \
                       detail::_PickDelete<T>(),          \
                       TypeIdentifier::Get<T>(),          \
                       #name),
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
#undef SCALAR_TYPE_META
      // Remaining entries default-initialize to empty TypeMetaData.
  };

  static std::once_flag init_once;
  std::call_once(init_once, [] {
    instances[static_cast<uint16_t>(c10::ScalarType::Undefined)] =
        detail::TypeMetaData{0,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr,
                             TypeIdentifier::Get<detail::_Uninitialized>(),
                             "Undefined"};
  });

  return instances;
}

uint16_t TypeMeta::existingMetaDataIndexForType(TypeIdentifier identifier) {
  auto* meta_datas = typeMetaDatas();
  const auto end = meta_datas + nextTypeIndex;
  auto it = std::find_if(meta_datas, end, [identifier](const auto& meta_data) {
    return meta_data.id_ == identifier;
  });
  if (it == end) {
    return kMaxTypeIndex;
  }
  return static_cast<uint16_t>(it - meta_datas);
}

CAFFE_DEFINE_KNOWN_TYPE(std::string, std_string)
CAFFE_DEFINE_KNOWN_TYPE(char, char)
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::mutex>, std_unique_ptr_std_mutex)
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>,
                        std_unique_ptr_std_atomic_bool)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int32_t>, std_vector_int32_t)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int64_t>, std_vector_int64_t)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<unsigned long>,  // NOLINT(runtime/int)
                        std_vector_unsigned_long)
CAFFE_DEFINE_KNOWN_TYPE(bool*, bool_ptr)
CAFFE_DEFINE_KNOWN_TYPE(char*, char_ptr)
CAFFE_DEFINE_KNOWN_TYPE(int*, int_ptr)
CAFFE_DEFINE_KNOWN_TYPE(
    detail::_guard_long_unique<long>,  // NOLINT(runtime/int)
    detail_guard_long_unique_long)
CAFFE_DEFINE_KNOWN_TYPE(
    detail::_guard_long_unique<std::vector<long>>,  // NOLINT(runtime/int)
    detail_guard_long_unique_std_vector_long)
CAFFE_DEFINE_KNOWN_TYPE(float*, float_ptr)
CAFFE_DEFINE_KNOWN_TYPE(at::Half*, at_Half)

}  // namespace caffe2
