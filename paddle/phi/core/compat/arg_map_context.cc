/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/arg_map_context.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/string_helper.h"

namespace phi {
std::ostream& operator<<(std::ostream& os, KernelSignature signature) {
  os << "Kernel Signature - name: " << signature.name << "; inputs: "
     << paddle::string::join_strings(std::get<0>(signature.args), ", ")
     << "; attributes: "
     << paddle::string::join_strings(std::get<1>(signature.args), ", ")
     << "; outputs: "
     << paddle::string::join_strings(std::get<2>(signature.args), ", ");
  return os;
}

}  // namespace phi
