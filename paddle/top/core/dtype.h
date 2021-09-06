/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

namespace pt {

/**
 * [ Why need new data type? ]
 *
 * The Var data type design in framework.proto is confusing, maybe we need
 * polish the VarType in framework.proto.
 *
 * We need to ensure that the operator library is relatively independent
 * and does not depend on the framework. Therefore, before calling the kernel
 * in the Tensor operation library inside the framework, the internal
 * data type needs to be converted to the data type in the Tensor operation
 * library.
 *
 */
enum class DataType {
  kUndef = 0,
  kBOOL,
  kINT8,   // Char
  kUINT8,  // BYte
  kINT16,
  kUINT16,
  kINT32,
  kINT64,
  kFLOAT16,
  kFLOAT32,
  kFLOAT64,
  kCOMPLEX64,
  kCOMPLEX128,
  kNumDataTypes,
};

}  // namespace pt
