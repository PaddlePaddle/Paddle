/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/raw_tensor.h"
#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

namespace phi {
template <>
const TypeInfo<phi::TensorBase>
    TypeInfoTraits<phi::TensorBase, paddle::framework::RawTensor>::kType =
        RegisterStaticType<phi::TensorBase>(
            paddle::framework::RawTensor::name());

template <>
const TypeInfo<phi::TensorBase>
    TypeInfoTraits<phi::TensorBase, paddle::framework::Vocab>::kType =
        RegisterStaticType<phi::TensorBase>(paddle::framework::Vocab::name());

template <>
const TypeInfo<phi::TensorBase>
    TypeInfoTraits<phi::TensorBase, paddle::framework::Strings>::kType =
        RegisterStaticType<phi::TensorBase>(paddle::framework::Strings::name());

template <>
const TypeInfo<phi::TensorBase>
    TypeInfoTraits<phi::TensorBase, paddle::framework::FeedList>::kType =
        RegisterStaticType<phi::TensorBase>(
            paddle::framework::FeedList::name());

template <>
const TypeInfo<phi::TensorBase>
    TypeInfoTraits<phi::TensorBase, egr::VariableCompatTensor>::kType =
        RegisterStaticType<phi::TensorBase>(egr::VariableCompatTensor::name());

template <>
const TypeInfo<phi::TensorBase>
    TypeInfoTraits<phi::TensorBase, paddle::prim::DescTensor>::kType =
        RegisterStaticType<phi::TensorBase>(paddle::prim::DescTensor::name());

}  // namespace phi
