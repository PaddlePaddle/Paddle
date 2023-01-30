/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/string_array.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

class Scope;

void SetFeedVariable(Scope* scope,
<<<<<<< HEAD
                     const phi::DenseTensor& input,
=======
                     const LoDTensor& input,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                     const std::string& var_name,
                     size_t index);

void SetFeedVariable(Scope* scope,
<<<<<<< HEAD
                     const std::vector<std::string>& input,
=======
                     const Strings& input,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                     const std::string& var_name,
                     size_t index);

FetchType& GetFetchVariable(const Scope& scope,
                            const std::string& var_name,
                            size_t index);

<<<<<<< HEAD
phi::DenseTensor& GetVariableTensor(const Scope& scope,
                                    const std::string& var_name);
=======
LoDTensor& GetVariableTensor(const Scope& scope, const std::string& var_name);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

}  // namespace framework
}  // namespace paddle
