//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace phi {

class Kernel;
class KernelKey;
class KernelArgsDef;
class KernelContext;
struct KernelSignature;
class ArgumentMappingContext;
class InferMetaContext;

using KernelFn = std::function<void(KernelContext* ctx)>;
using KernelArgsDefFn = void (*)(const KernelKey& kernel_key, Kernel* kernel);
using KernelArgsParseFn = void (*)(const KernelKey& default_key,
                                   KernelArgsDef* args_def);

using ArgumentMappingFn =
    std::function<KernelSignature(const ArgumentMappingContext&)>;
using InferMetaFn = void (*)(InferMetaContext* ctx);

// Global SmallVector size setting
constexpr size_t kInputSmallVectorSize = 15U;
constexpr size_t kAttrSmallVectorSize = 15U;
constexpr size_t kOutputSmallVectorSize = 15U;

}  // namespace phi
