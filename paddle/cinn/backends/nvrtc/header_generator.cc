// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/nvrtc/header_generator.h"

#include <fstream>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/cinn/common/common.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {
namespace nvrtc {

HeaderGeneratorBase& JitSafeHeaderGenerator::GetInstance() {
  static JitSafeHeaderGenerator instance;
  return instance;
}

const size_t JitSafeHeaderGenerator::size() const {
  PADDLE_ENFORCE_EQ(include_names_.size(),
                    headers_.size(),
                    ::common::errors::InvalidArgument(
                        "Internal error in size of header files."));
  return include_names_.size();
}

std::string read_file_as_string(const std::string& file_path) {
#ifdef RUNTIME_INCLUDE_DIR
  static constexpr char* defined_runtime_include_dir = RUNTIME_INCLUDE_DIR;
#else
  static constexpr char* defined_runtime_include_dir = nullptr;
#endif

#ifdef CINN_WITH_CUDA
  std::string cinn_path =
      defined_runtime_include_dir ? defined_runtime_include_dir : "";
  std::ifstream file(cinn_path + '/' + file_path);

  if (!file.is_open()) {
    VLOG(1) << "Unable to open file : " << cinn_path << '/' << file_path;
    return "";
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  return buffer.str();
#else
  return "";
#endif
}
#ifdef CINN_WITH_CUDA

static const std::string cinn_float16_header =  // NOLINT
    read_file_as_string("float16.h");
static const std::string cinn_bfloat16_header =  // NOLINT
    read_file_as_string("bfloat16.h");
static const std::string cinn_with_cuda_header =  // NOLINT
    R"(
#pragma once
#define CINN_WITH_CUDA
)";
static const std::string cinn_cuda_runtime_source_header =  // NOLINT
    read_file_as_string("cinn_cuda_runtime_source.cuh");
#endif
JitSafeHeaderGenerator::JitSafeHeaderGenerator() {
  const auto& headers_map = ::jitify::detail::get_jitsafe_headers_map();
  for (auto& pair : headers_map) {
    include_names_.emplace_back(pair.first.data());
    headers_.emplace_back(pair.second.data());
  }
#ifdef CINN_WITH_CUDA
  include_names_.emplace_back("float16_h");
  headers_.emplace_back(cinn_float16_header.data());
  include_names_.emplace_back("bfloat16_h");
  headers_.emplace_back(cinn_bfloat16_header.data());
  include_names_.emplace_back("cinn_with_cuda_h");
  headers_.emplace_back(cinn_with_cuda_header.data());
  include_names_.emplace_back("cinn_cuda_runtime_source_h");
  headers_.emplace_back(cinn_cuda_runtime_source_header.data());
#endif
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
