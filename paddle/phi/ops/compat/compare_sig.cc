// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature LessThanArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("less_than", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature LessEqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("less_equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature GreaterThanArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("greater_than", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature GreaterEqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("greater_equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature EqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature NotEqualArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("not_equal", {"X", "Y"}, {"axis"}, {"Out"});
}

KernelSignature EqualAllArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("equal_all", {"X", "Y"}, {}, {"Out"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(less_than, phi::LessThanArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(less_equal, phi::LessEqualArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(greater_than, phi::GreaterThanArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(greater_equal, phi::GreaterEqualArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(equal, phi::EqualArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(not_equal, phi::NotEqualArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(equal_all, phi::EqualAllArgumentMapping);
