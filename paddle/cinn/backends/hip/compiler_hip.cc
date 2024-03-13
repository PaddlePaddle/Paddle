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

#include "paddle/cinn/backends/hip/compiler_hip.h"
#include "paddle/cinn/backends/nvrtc/header_generator.h"
#include "paddle/cinn/runtime/hip/hip_backend_api.h"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "paddle/cinn/backends/cuda_util.h"

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace backends {
namespace hiprtc {

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  return CompileHipSource(code, include_headers);
}

Compiler::Compiler() {
}

std::vector<std::string> Compiler::FindHIPIncludePaths() {
    const std::string delimiter = "/";
    std::string cuda_include_path;
    const char* cuda_path_env = std::getenv("ROCM_PATH");//??? rocm_path ??? hip_path
    if (cuda_path_env != nullptr) {
        cuda_include_path += cuda_path_env;
        cuda_include_path += delimiter + "include";
        return {cuda_include_path};
    }

#if defined(__linux__)
    struct stat st;
    cuda_include_path = "/usr/local/cuda/include";
    if (stat(cuda_include_path.c_str(), &st) == 0) {
        return {cuda_include_path};
    }
#endif
    LOG(FATAL) << "Cannot find hip include path."
                << "HIP_PATH is not set or HIP is not installed in the default "
                    "installation path."
                << "In other than linux, it is necessary to set HIP_PATH.";
    return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
    return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileHipSource(const std::string& code,
                                        bool include_headers) {
    const auto& header_gen = cinn::backends::nvrtc::JitSafeHeaderGenerator::GetInstance();//?
    std::vector<std::string> compile_options;
    std::vector<const char*> param_cstrings{};
    hiprtcProgram prog;
    std::string cc = "30";//?
    int major, minor;
    hipError_t e1 =
        hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor , 0);
    hipError_t e2 =
        hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, 0);
    if (e1 == hipSuccess && e2 == hipSuccess) {
        cc = std::to_string(major) + std::to_string(minor);
    } else {
        LOG(WARNING) << "cannot detect compute capability from your device, "
                    << "fall back to compute_30.";
    }
    
    compile_options.push_back("-arch=compute_" + cc);//hu shi compile to cubin
    compile_options.push_back("-std=c++14");
    compile_options.push_back("-default-device");

    if (include_headers) {  // prepare include headers
    auto cuda_headers = FindHIPIncludePaths();
    auto cinn_headers = FindCINNRuntimeIncludePaths();
    std::vector<std::string> include_paths;
    for (auto& header : cuda_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    for (auto& header : cinn_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    compile_options.insert(
        std::end(compile_options), include_paths.begin(), include_paths.end());
    }

    for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
    }

    VLOG(3) << "compile options: " << utils::Join(compile_options, " ");
    //HIP_CALL(
        hiprtcCreateProgram(&prog,
                                    code.c_str(),
                                    nullptr,
                                    header_gen.size(),
                                    const_cast<const char**>(header_gen.headers().data()),
                                    const_cast<const char**>(header_gen.include_names().data()));

    hiprtcResult compile_res = hiprtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

    {  // get log
    size_t log_size;
    //HIP_CALL(
    hiprtcGetProgramLogSize(prog, &log_size);
    std::string log;
    log.resize(log_size);
    //HIP_CALL
    hiprtcGetProgramLog(prog, &log[0]);
    CHECK_EQ(compile_res, HIPRTC_SUCCESS) << log;
    }

    size_t size;
    std::string data;
    //HIP_CALL(
    hiprtcGetCodeSize(prog, &size);
    data.resize(size);
    //HIP_CALL(
    hiprtcGetCode(prog, &data[0]);

    hiprtcDestroyProgram(&prog);
    return data;
}

std::string Compiler::ReadFile(const std::string& file_name,
                               std::ios_base::openmode mode) {
  // open cubin file
  std::ifstream ifs(file_name, mode);
  CHECK(ifs.is_open()) << "Fail to open file " << file_name;
  ifs.seekg(std::ios::end);
  auto len = ifs.tellg();
  ifs.seekg(0);

  // read cubin file
  std::string file_data(len, ' ');
  ifs.read(&file_data[0], len);
  ifs.close();
  return std::move(file_data);
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
