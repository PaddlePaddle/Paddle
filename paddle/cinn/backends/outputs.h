// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <string>

namespace cinn {
namespace backends {

/**
 * A struct specifying a collection of outputs.
 */
struct Outputs {
  //! The name of the emitted object file. Empty if no object file is desired.
  std::string object_name;

  //! The name of the emitted llvm bitcode. Empty if no bitcode file is desired.
  std::string bitcode_name;

  //! The name of the emitted C header file.
  std::string c_header_name;

  //! The name of the emitted C source file.
  std::string c_source_name;

  //! The name of the emitted CUDA source file.
  std::string cuda_source_name;

  Outputs object(const std::string& name) const;

  Outputs bitcode(const std::string& name) const;

  Outputs c_header(const std::string& name) const;

  Outputs c_source(const std::string& name) const;

  Outputs cuda_source(const std::string& name) const;
};

}  // namespace backends
}  // namespace cinn
