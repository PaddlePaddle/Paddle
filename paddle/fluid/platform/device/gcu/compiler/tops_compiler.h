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

#pragma once

#include <tops/tops_ext.h>
#include <memory>

namespace hlir {
class Module;
}

namespace paddle {
namespace platform {
namespace gcu {

topsExecutable_t CompileExecutable(std::shared_ptr<hlir::Module> module);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
