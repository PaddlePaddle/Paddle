// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/hip/compiler_hip.h"

#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>
#include <fstream>
#include <iostream>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/runtime/hip/hip_util.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace backends {
namespace hiprtc {

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  if (runtime::UseHipccCompiler()) {
    return CompileWithHipcc(code);
  }
  return CompileWithHiprtc(code, include_headers);
}

std::vector<std::string> Compiler::FindHIPIncludePaths() {
  const std::string delimiter = "/";
  std::string hip_include_path;
  const char* hip_path_env = std::getenv("ROCM_PATH");
  if (hip_path_env != nullptr) {
    hip_include_path += hip_path_env;
    hip_include_path += delimiter + "include";
    return {hip_include_path};
  }

#if defined(__linux__)
  struct stat st;
  hip_include_path = "/opt/rocm/include";
  if (stat(hip_include_path.c_str(), &st) == 0) {
    return {hip_include_path};
  }
#endif
  PADDLE_THROW(::common::errors::Fatal(
      "Cannot find hip include path. ROCM_PATH is not set or HIP is not "
      "installed in the default installation path. In other than linux, it is "
      "necessary to set ROCM_PATH."));
  return {hip_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileWithHiprtc(const std::string& code,
                                        bool include_headers) {
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  hiprtcProgram prog;
  compile_options.push_back(std::string("--gpu-architecture=") +
                            GetDeviceArch());
  compile_options.push_back("-std=c++17");

  // prepare include headers
  std::vector<std::string> hip_headers = FindHIPIncludePaths();
  std::vector<std::string> cinn_headers = FindCINNRuntimeIncludePaths();
  std::vector<std::string> include_paths;
  for (const auto& header : hip_headers) {
    include_paths.push_back("--include-path=" + header);
  }
  for (const auto& header : cinn_headers) {
    include_paths.push_back("--include-path=" + header);
  }
  compile_options.insert(
      std::end(compile_options), include_paths.begin(), include_paths.end());

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }
  VLOG(5) << "hip compile options: " << utils::Join(compile_options, " ");
  HIPRTC_CHECK(
      hiprtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  hiprtcResult compile_res =
      hiprtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  {
    // check compile result and get log
    size_t log_size;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    PADDLE_ENFORCE_EQ(
        compile_res,
        HIPRTC_SUCCESS,
        ::common::errors::External("HIPRTC Error in Paddle CINN: %s", log));
  }

  size_t size;
  std::string data;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &size));
  data.resize(size);
  HIPRTC_CHECK(hiprtcGetCode(prog, &data[0]));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return data;
}

std::string Compiler::CompileWithHipcc(const std::string& hip_c) {
  // hipcc compile command
  std::string options = "hipcc -O3 --genco";
  // device arch
  options += std::string(" --offload-arch=") + GetDeviceArch();

  std::vector<std::string> include_dir = FindCINNRuntimeIncludePaths();
  std::string include_dir_str = "";
  for (const auto& dir : include_dir) {
    if (include_dir_str.empty()) {
      include_dir_str = dir;
    } else {
      include_dir_str += ":" + dir;
    }
  }

  std::string dir = "./source";
  // create the folder to store sycl temporary files
  if (access(dir.c_str(), F_OK) == -1) {
    PADDLE_ENFORCE_NE(mkdir(dir.c_str(), 7),
                      -1,
                      ::common::errors::PreconditionNotMet(
                          "Fail to mkdir %s in Hipcc compile.", dir));
  }
  prefix_name_ = dir + "/" + common::UniqName("hip_tmp");

  std::string hip_c_file = prefix_name_ + ".cc";
  std::ofstream ofs(hip_c_file, std::ios::out);
  PADDLE_ENFORCE_EQ(ofs.is_open(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Fail to open file %s to compile HIP.", hip_c_file));
  ofs << hip_c;
  ofs.close();

  options += " -I " + include_dir_str;
  options += " -o " + prefix_name_ + ".hsaco";
  options += " " + prefix_name_ + ".cc";
  VLOG(5) << "hip compile options: " << options;
  system(options.c_str());
  return prefix_name_ + ".hsaco";
}

std::string Compiler::GetDeviceArch() {
  // Get device properties from the first device available.
  hipDeviceProp_t props;
  constexpr unsigned int device_id = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));
  return props.gcnArchName;
}

}  // namespace hiprtc
}  // namespace backends
}  // namespace cinn
