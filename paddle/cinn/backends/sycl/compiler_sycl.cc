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

#include "paddle/cinn/backends/sycl/compiler_sycl.h"
#include <sys/stat.h> //for mkdir ...
#include <fstream>

namespace cinn {
namespace backends {
namespace syclrtc {

Compiler::Compiler() {}

std::string Compiler::operator()(const std::string& code, const Target::Arch gpu_type) {
  return CompileToSo(code, gpu_type);
}


class NUM {
  public:
    static int n;
    static int getNum(){
      return n++;
    }
};
int NUM::n = 0;

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileToSo(const std::string& source_code, const Target::Arch gpu_type) {
  // read dir source
  std::string dir = "./source";
  if (access(dir.c_str(), 0) == -1) {
    CHECK(mkdir(dir.c_str(), 7) != -1) << "Fail to mkdir " << dir;
  }

  // get unqiue prefix name
  // std::string prefix_name                    = dir + "/" + common::UniqName("rtc_tmp");
  std::string prefix_name                    = dir + "/" + std::to_string(NUM::getNum());
  std::string sycl_source_file    = prefix_name + ".cpp";
  std::string sycl_shared_library = prefix_name + ".so";
  std::ofstream ofs(sycl_source_file.c_str(), std::ios::out);
  CHECK(ofs.is_open()) << "Fail to open file " << sycl_source_file;
  ofs << source_code;
  ofs.close();

  // compile options
  std::string options = SYCL_CXX_COMPILER;
  // prepare include headers
  auto cinn_headers = FindCINNRuntimeIncludePaths();
  for (auto& header : cinn_headers) {
    options += " -I " + header;
  }
  options += " -fsycl";
  options += " -fsycl-targets=";
  if (gpu_type == Target::Arch::NVGPU) {
    options += "nvptx64-nvidia-cuda";
  } else if (gpu_type == Target::Arch::AMDGPU) {
    options += "amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908";
  } else if (gpu_type == Target::Arch::IntelGPU) {
    options += "";
  }
  options += " -std=c++17";
  options += " -lpthread";
  options += " -fPIC -shared";
  options += " -ldl";
  options += " -O3";
  options += " -ffast-math";
  options += " -fbracket-depth=1030"; // set 1030 for constant op, default max bracket-depth = 256
  options += " -o " + sycl_shared_library;
  options += " " + sycl_source_file;
  VLOG(2) << "SYCL Compile Options : " << options;

  // compile
  std::cout << options << std::endl;
  CHECK(system(options.c_str()) == 0) << options;
  return sycl_shared_library;
}

std::string Compiler::GetDeviceArch() {}

std::string Compiler::ReadFile(const std::string& file_path, std::ios_base::openmode mode) {}

}  // namespace syclrtc
}  // namespace backends
}  // namespace cinn