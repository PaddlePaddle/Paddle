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

namespace pten {

class Kernel;
class KernelKey;
class KernelArgsDef;
class KernelContext;

using KernelFn = void (*)(KernelContext* ctx);
using KernelArgsDefFn = void (*)(Kernel* kernel);
using KernelArgsParseFn = void (*)(const KernelKey& default_key,
                                   KernelArgsDef* args_def);

// Multiple kernels of the same operation are distinguished by the difference
// of the overload name. For the convenience of reuse, we define some overload
// naming strings for the naming of the kernel

// For kernels that contains dynamic tensor attribute and it need to be always
// on host device, such as `ScaleTensor`
constexpr char kContainHostTensorSuffix[] = "host";

// For kernels with SelectedRowsTensor input and output
constexpr char kContainSelectedRowsSuffix[] = "sr";

// For kernels with intermediate output
constexpr char kContainMidOutputTensorSuffix[] = "mid";
}  // namespace pten
