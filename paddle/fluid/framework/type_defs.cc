/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/type_defs.h"

#include "paddle/common/enforce.h"

namespace paddle {

using namespace framework;  // NOLINT
template class variant<paddle::blank,
                       int,
                       float,
                       std::string,
                       std::vector<int>,
                       std::vector<float>,
                       std::vector<std::string>,
                       bool,
                       std::vector<bool>,
                       BlockDesc*,
                       int64_t,
                       std::vector<BlockDesc*>,
                       std::vector<int64_t>,
                       std::vector<double>,
                       VarDesc*,
                       std::vector<VarDesc*>,
                       double,
                       paddle::experimental::Scalar,
                       std::vector<paddle::experimental::Scalar>,
                       ::pir::Block*,
                       std::vector<::pir::Value>,
                       std::shared_ptr<::pir::Program>>;
}  // namespace paddle
REGISTER_LOG_SIMPLY_STR(paddle::framework::AttributeMap);
REGISTER_LOG_SIMPLY_STR(paddle::framework::Attribute);
