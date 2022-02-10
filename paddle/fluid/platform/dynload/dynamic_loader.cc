/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/pten/backends/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

void SetPaddleLibPath(const std::string& py_site_pkg_path) {
  pten::dynload::SetPaddleLibPath(py_site_pkg_path);
}

void* GetCublasDsoHandle() { return pten::dynload::GetCublasDsoHandle(); }

void* GetCublasLtDsoHandle() { return pten::dynload::GetCublasLtDsoHandle(); }

void* GetCUDNNDsoHandle() { return pten::dynload::GetCUDNNDsoHandle(); }

void* GetCUPTIDsoHandle() { return pten::dynload::GetCUPTIDsoHandle(); }

void* GetCurandDsoHandle() { return pten::dynload::GetCurandDsoHandle(); }

#ifdef PADDLE_WITH_HIP
void* GetROCFFTDsoHandle() { return pten::dynload::GetROCFFTDsoHandle(); }
#endif

void* GetNvjpegDsoHandle() { return pten::dynload::GetNvjpegDsoHandle(); }

void* GetCusolverDsoHandle() { return pten::dynload::GetCusolverDsoHandle(); }

void* GetCusparseDsoHandle() { return pten::dynload::GetCusparseDsoHandle(); }

void* GetNVRTCDsoHandle() { return pten::dynload::GetNVRTCDsoHandle(); }

void* GetCUDADsoHandle() { return pten::dynload::GetCUDADsoHandle(); }

void* GetWarpCTCDsoHandle() { return pten::dynload::GetWarpCTCDsoHandle(); }

void* GetNCCLDsoHandle() { return pten::dynload::GetNCCLDsoHandle(); }
void* GetHCCLDsoHandle() { return pten::dynload::GetHCCLDsoHandle(); }

void* GetTensorRtDsoHandle() { return pten::dynload::GetTensorRtDsoHandle(); }

void* GetMKLMLDsoHandle() { return pten::dynload::GetMKLMLDsoHandle(); }

void* GetLAPACKDsoHandle() { return pten::dynload::GetLAPACKDsoHandle(); }

void* GetOpDsoHandle(const std::string& dso_name) {
  return pten::dynload::GetOpDsoHandle(dso_name);
}

void* GetNvtxDsoHandle() { return pten::dynload::GetNvtxDsoHandle(); }

void* GetCUFFTDsoHandle() { return pten::dynload::GetCUFFTDsoHandle(); }

void* GetMKLRTDsoHandle() { return pten::dynload::GetMKLRTDsoHandle(); }

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
