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

#include "paddle/tcmpt/core/dtype.h"

namespace pt {

std::ostream& operator<<(std::ostream& os, DataType dtype) {
  switch (dtype) {
    case DataType::kUndef:
      os << "Undefined";
      break;
    case DataType::kBOOL:
      os << "bool";
      break;
    case DataType::kINT8:
      os << "int8";
      break;
    case DataType::kUINT8:
      os << "uint8";
      break;
    case DataType::kINT16:
      os << "int16";
      break;
    case DataType::kINT32:
      os << "int32";
      break;
    case DataType::kINT64:
      os << "int64";
      break;
    case DataType::kFLOAT16:
      os << "float16";
      break;
    case DataType::kFLOAT32:
      os << "float32";
      break;
    case DataType::kFLOAT64:
      os << "float64";
      break;
    case DataType::kCOMPLEX64:
      os << "complex64";
      break;
    case DataType::kCOMPLEX128:
      os << "complex128";
      break;
    default:
      // TODO(chenweihang): change to enforce later
      throw std::runtime_error("Invalid DataType type.");
  }
  return os;
}

}  // namespace pt
