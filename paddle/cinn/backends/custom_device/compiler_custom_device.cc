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

#include "paddle/cinn/backends/custom_device/compiler_custom_device.h"

#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>
#include <fstream>
#include <iostream>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/custom_device/custom_device_util.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace backends {
namespace cdrtc {

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  if (runtime::UseCdccCompiler()) {
    return CompileWithCdcc(code);
  }
  return CompileWithCdrtc(code, include_headers);
}

std::vector<std::string> Compiler::FindCustomDeviceIncludePaths() {
  const std::string delimiter = "/";
  std::string custom_device_include_path;
  const char* custom_device_path_env = std::getenv("ROCM_PATH");
  if (custom_device_path_env != nullptr) {
    custom_device_include_path += custom_device_path_env;
    custom_device_include_path += delimiter + "include";
    return {custom_device_include_path};
  }

#if defined(__linux__)
  struct stat st;
  custom_device_include_path = "/opt/rocm/include";
  if (stat(custom_device_include_path.c_str(), &st) == 0) {
    return {custom_device_include_path};
  }
#endif
  PADDLE_THROW(::common::errors::Fatal(
      "Cannot find custom_device include path. ROCM_PATH is not set or "
      "CUSTOMDEVICE is not "
      "installed in the default installation path. In other than linux, it is "
      "necessary to set ROCM_PATH."));
  return {custom_device_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileWithCdrtc(const std::string& code,
                                       bool include_headers) {
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  cdrtcProgram prog;
  compile_options.push_back(std::string("--gpu-architecture=") +
                            GetDeviceArch());
  compile_options.push_back("-std=c++17");

  // prepare include headers
  std::vector<std::string> custom_device_headers =
      FindCustomDeviceIncludePaths();
  std::vector<std::string> cinn_headers = FindCINNRuntimeIncludePaths();
  std::vector<std::string> include_paths;
  for (const auto& header : custom_device_headers) {
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
  VLOG(5) << "custom_device compile options: "
          << utils::Join(compile_options, " ");
  CDRTC_CHECK(
      cdrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  cdrtcResult compile_res =
      cdrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  {
    // check compile result and get log
    size_t log_size;
    CDRTC_CHECK(cdrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    CDRTC_CHECK(cdrtcGetProgramLog(prog, &log[0]));
    PADDLE_ENFORCE_EQ(
        compile_res,
        CDRTC_SUCCESS,
        ::common::errors::External("CDRTC Error in Paddle CINN: %s", log));
  }

  size_t size;
  std::string data;
  CDRTC_CHECK(cdrtcGetCodeSize(prog, &size));
  data.resize(size);
  CDRTC_CHECK(cdrtcGetCode(prog, &data[0]));
  CDRTC_CHECK(cdrtcDestroyProgram(&prog));
  return data;
}

std::string Compiler::CompileWithCdcc(const std::string& custom_device_c) {
  // custom_devicecc compile command
  std::string options = "custom_devicecc -O3 --genco";  // TODO(xuyuhan)
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
                          "Fail to mkdir %s in Cdcc compile.", dir));
  }
  prefix_name_ = dir + "/" + common::UniqName("custom_device_tmp");

  std::string custom_device_c_file = prefix_name_ + ".cc";
  std::ofstream ofs(custom_device_c_file, std::ios::out);
  PADDLE_ENFORCE_EQ(ofs.is_open(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Fail to open file %s to compile CUSTOMDEVICE.",
                        custom_device_c_file));
  ofs << custom_device_c;
  ofs.close();

  options += " -I " + include_dir_str;
  options += " -o " + prefix_name_ + ".hsaco";
  options += " " + prefix_name_ + ".cc";
  VLOG(5) << "custom_device compile options: " << options;
  system(options.c_str());
  return prefix_name_ + ".hsaco";
}

std::string Compiler::GetDeviceArch() {
  // Get device properties from the first device available.
  custom_deviceDeviceProp_t props;
  constexpr unsigned int device_id = 0;
  CUSTOMDEVICE_CHECK(customDeviceGetDeviceProperties(&props, device_id));
  return props.gcnArchName;
}

}  // namespace cdrtc
}  // namespace backends
}  // namespace cinn
