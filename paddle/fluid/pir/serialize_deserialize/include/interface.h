// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef PADDLE_FLUID_PIR_SERIALIZE_DESERIALIZE_INCLUDE_INTERFACE_H_
#define PADDLE_FLUID_PIR_SERIALIZE_DESERIALIZE_INCLUDE_INTERFACE_H_
#include "paddle/pir/include/core/program.h"
namespace pir {

void WriteModule(const pir::Program& program,
                 const std::string& file_path,
                 const uint64_t& pir_version,
                 bool overwrite,
                 bool readable);

void ReadModule(const std::string& file_path, pir::Program* program);

}  // namespace pir
#endif  // PADDLE_FLUID_PIR_SERIALIZE_DESERIALIZE_INCLUDE_INTERFACE_H_
