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

DEFINE_string(cudnn_dir, "",
              "Specify path for loading libcudnn.so. For instance, "
              "/usr/local/cudnn/lib. If empty [default], dlopen "
              "will search cudnn from LD_LIBRARY_PATH");

DEFINE_string(
    cuda_dir, "",
    "Specify path for loading cuda library, such as libcublas, libcublasLt "
    "libcurand, libcusolver. For instance, /usr/local/cuda/lib64. "
    "If default, dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(nccl_dir, "",
              "Specify path for loading nccl library, such as libnccl.so. "
              "For instance, /usr/local/cuda/lib64. If default, "
              "dlopen will search cuda from LD_LIBRARY_PATH");

DEFINE_string(hccl_dir, "",
              "Specify path for loading hccl library, such as libhccl.so. "
              "For instance, "
              "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/. If "
              "default, "
              "dlopen will search hccl from LD_LIBRARY_PATH");

DEFINE_string(cupti_dir, "", "Specify path for loading cupti.so.");

DEFINE_string(
    tensorrt_dir, "",
    "Specify path for loading tensorrt library, such as libnvinfer.so.");

DEFINE_string(mklml_dir, "", "Specify path for loading libmklml_intel.so.");

DEFINE_string(lapack_dir, "", "Specify path for loading liblapack.so.");

DEFINE_string(mkl_dir, "",
              "Specify path for loading libmkl_rt.so. "
              "For insrance, /opt/intel/oneapi/mkl/latest/lib/intel64/."
              "If default, "
              "dlopen will search mkl from LD_LIBRARY_PATH");

DEFINE_string(op_dir, "", "Specify path for loading user-defined op library.");

#ifdef PADDLE_WITH_HIP

DEFINE_string(miopen_dir, "",
              "Specify path for loading libMIOpen.so. For instance, "
              "/opt/rocm/miopen/lib. If empty [default], dlopen "
              "will search miopen from LD_LIBRARY_PATH");

DEFINE_string(rocm_dir, "",
              "Specify path for loading rocm library, such as librocblas, "
              "libmiopen, libhipsparse. For instance, /opt/rocm/lib. "
              "If default, dlopen will search rocm from LD_LIBRARY_PATH");

DEFINE_string(rccl_dir, "",
              "Specify path for loading rccl library, such as librccl.so. "
              "For instance, /opt/rocm/rccl/lib. If default, "
              "dlopen will search rccl from LD_LIBRARY_PATH");
#endif

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
