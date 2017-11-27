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
#include <stdlib.h>
#include <string.h>

namespace paddle {
namespace platform {

static std::vector<std::string> kCudaProfileConfiguration = {
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "dynsmemperblock",
    "stasmemperblock",
    "regperthread",
    "memtransfersize",
    "memtransferdir",
    "memtransferhostmemtype",
    "streamid",
    "cacheconfigrequested",
    "cacheconfigexecuted",
    "countermodeaggregate",
    "enableonstart 0",
    "active_warps",
    "active_cycles",
};

void CudaProfilerInit(std::string output_file, std::string output_mode) {
  std::array<char, 128> buf;
  std::string tmpl = "/tmp/cuda_profile_config.XXXXXX";
  PADDLE_ENFORCE_LT(tmpl.size(), buf.size());
  memcpy(buf.data(), tmpl.data(), tmpl.size());
  auto result = mktemp(buf.data());
  PADDLE_ENFORCE(strlen(result) != 0);
  std::string config = result;

  {
    std::ofstream ofs(config, std::ios::out | std::ios::trunc);
    PADDLE_ENFORCE(ofs.is_open(), "ofstream: ", ofs.rdstate());
    for (const auto& line : kCudaProfileConfiguration) {
      ofs << line << std::endl;
    }
  }

  PADDLE_ENFORCE(output_mode == "key_value" || output_mode == "csv");
  cudaOutputMode_t mode = output_mode == "csv" ? cudaCSV : cudaKeyValuePair;
  PADDLE_ENFORCE(
      cudaProfilerInitialize(config.c_str(), output_file.c_str(), mode));
}

void CudaProfilerStart() { PADDLE_ENFORCE(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE((cudaProfilerStop())); }
}
}
