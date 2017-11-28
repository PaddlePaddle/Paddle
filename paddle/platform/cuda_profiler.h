/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace paddle {
namespace platform {

void CudaProfilerInit(std::string output_file, std::string output_mode,
                      std::vector<std::string> config_flags) {
  std::array<char, 128> buf;
  std::string tmpl = "/tmp/cuda_profile_config.XXXXXX";
  PADDLE_ENFORCE_LT(tmpl.size(), buf.size());
  memcpy(buf.data(), tmpl.data(), tmpl.size());
  auto result = mktemp(buf.data());
  PADDLE_ENFORCE(strlen(result) != 0);
  std::string config_file = result;

  {
    std::ofstream ofs(config_file, std::ios::out | std::ios::trunc);
    PADDLE_ENFORCE(ofs.is_open(), "ofstream: ", ofs.rdstate());
    for (const auto& line : config_flags) {
      ofs << line << std::endl;
    }
  }

  PADDLE_ENFORCE(output_mode == "kvp" || output_mode == "csv");
  cudaOutputMode_t mode = output_mode == "csv" ? cudaCSV : cudaKeyValuePair;
  PADDLE_ENFORCE(
      cudaProfilerInitialize(config_file.c_str(), output_file.c_str(), mode));
}

void CudaProfilerStart() { PADDLE_ENFORCE(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE(cudaProfilerStop()); }

}  // namespace platform
}  // namespace paddle
