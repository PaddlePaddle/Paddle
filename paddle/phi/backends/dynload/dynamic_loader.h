/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include <string>
#include "paddle/utils/test_macros.h"
namespace phi {
namespace dynload {

#ifndef _WIN32
#define DECLARE_TYPE(__name, ...) decltype(__name(__VA_ARGS__))
#else
#define DECLARE_TYPE(__name, ...) decltype(auto)
#endif

void* GetCublasDsoHandle();
void* GetCublasLtDsoHandle();
TEST_API void* GetCUDNNDsoHandle();
void* GetCUPTIDsoHandle();
void* GetCurandDsoHandle();
void* GetNvjpegDsoHandle();
void* GetCusolverDsoHandle();
void* GetCusparseDsoHandle();
void* GetNVRTCDsoHandle();
void* GetCUDADsoHandle();
void* GetWarpCTCDsoHandle();
void* GetWarpRNNTDsoHandle();
void* GetFlashAttnDsoHandle();
void* GetFlashAttnV3DsoHandle();
void* GetNCCLDsoHandle();
void* GetTensorRtDsoHandle();
void* GetMKLMLDsoHandle();
void* GetLAPACKDsoHandle();
void* GetOpDsoHandle(const std::string& dso_name);
void* GetNvtxDsoHandle();
void* GetCUFFTDsoHandle();
void* GetMKLRTDsoHandle();
void* GetROCFFTDsoHandle();
void* GetCusparseLtDsoHandle();
void* GetXPTIDsoHandle();
void* GetAfsApiDsoHandle();

void SetPaddleLibPath(const std::string&);

}  // namespace dynload
}  // namespace phi
