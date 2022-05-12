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
#include "paddle/phi/backends/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

void SetPaddleLibPath(const std::string& py_site_pkg_path) {
  phi::dynload::SetPaddleLibPath(py_site_pkg_path);
}

void* GetCublasDsoHandle() { return phi::dynload::GetCublasDsoHandle(); }

void* GetCublasLtDsoHandle() { return phi::dynload::GetCublasLtDsoHandle(); }

void* GetCUDNNDsoHandle() { return phi::dynload::GetCUDNNDsoHandle(); }

void* GetCUPTIDsoHandle() { return phi::dynload::GetCUPTIDsoHandle(); }

void* GetCurandDsoHandle() { return phi::dynload::GetCurandDsoHandle(); }

#ifdef PADDLE_WITH_HIP
void* GetROCFFTDsoHandle() { return phi::dynload::GetROCFFTDsoHandle(); }
#endif

void* GetNvjpegDsoHandle() { return phi::dynload::GetNvjpegDsoHandle(); }

void* GetCusolverDsoHandle() { return phi::dynload::GetCusolverDsoHandle(); }

void* GetCusparseDsoHandle() { return phi::dynload::GetCusparseDsoHandle(); }

void* GetNVRTCDsoHandle() { return phi::dynload::GetNVRTCDsoHandle(); }

void* GetCUDADsoHandle() { return phi::dynload::GetCUDADsoHandle(); }

void* GetWarpCTCDsoHandle() { return phi::dynload::GetWarpCTCDsoHandle(); }

void* GetNCCLDsoHandle() { return phi::dynload::GetNCCLDsoHandle(); }
void* GetHCCLDsoHandle() { return phi::dynload::GetHCCLDsoHandle(); }

void* GetTensorRtDsoHandle() { return phi::dynload::GetTensorRtDsoHandle(); }

void* GetMKLMLDsoHandle() { return phi::dynload::GetMKLMLDsoHandle(); }

void* GetLAPACKDsoHandle() { return phi::dynload::GetLAPACKDsoHandle(); }

void* GetOpDsoHandle(const std::string& dso_name) {
  return phi::dynload::GetOpDsoHandle(dso_name);
}

void* GetNvtxDsoHandle() { return phi::dynload::GetNvtxDsoHandle(); }

void* GetCUFFTDsoHandle() { return phi::dynload::GetCUFFTDsoHandle(); }

void* GetMKLRTDsoHandle() { return phi::dynload::GetMKLRTDsoHandle(); }

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
