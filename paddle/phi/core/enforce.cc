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

#include "paddle/phi/core/enforce.h"

#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/blank.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/external_error.pb.h"
#endif  // PADDLE_WITH_CUDA

COMMON_DECLARE_int32(call_stack_level);
namespace phi::enforce {

void ThrowWarnInternal(const std::string& msg) {
  LOG(WARNING) << "WARNING :" << msg;
}

/**************************************************************************/
/**************************** NVIDIA ERROR ********************************/
#ifdef PADDLE_WITH_CUDA

}  // namespace phi::enforce
namespace phi::enforce::details {

template <typename T>
struct ExternalApiProtoType {};

#define DEFINE_EXTERNAL_API_PROTO_TYPE(type, proto_type)    \
  template <>                                               \
  struct ExternalApiProtoType<type> {                       \
    using Type = type;                                      \
    static constexpr const char* kTypeString = #proto_type; \
    static constexpr phi::proto::ApiType kProtoType =       \
        phi::proto::ApiType::proto_type;                    \
  }

DEFINE_EXTERNAL_API_PROTO_TYPE(cudaError_t, CUDA);
DEFINE_EXTERNAL_API_PROTO_TYPE(curandStatus_t, CURAND);
DEFINE_EXTERNAL_API_PROTO_TYPE(cudnnStatus_t, CUDNN);
DEFINE_EXTERNAL_API_PROTO_TYPE(cublasStatus_t, CUBLAS);
DEFINE_EXTERNAL_API_PROTO_TYPE(cusparseStatus_t, CUSPARSE);
DEFINE_EXTERNAL_API_PROTO_TYPE(cusolverStatus_t, CUSOLVER);
DEFINE_EXTERNAL_API_PROTO_TYPE(cufftResult_t, CUFFT);
DEFINE_EXTERNAL_API_PROTO_TYPE(CUresult, CU);

#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
DEFINE_EXTERNAL_API_PROTO_TYPE(ncclResult_t, NCCL);
#endif

#undef DEFINE_EXTERNAL_API_PROTO_TYPE

}  // namespace phi::enforce::details
namespace phi::enforce {

template <typename T>
inline const char* GetErrorMsgUrl(T status) {
  using __CUDA_STATUS_TYPE__ = decltype(status);
  phi::proto::ApiType proto_type =
      details::ExternalApiProtoType<__CUDA_STATUS_TYPE__>::kProtoType;
  switch (proto_type) {
    case phi::proto::ApiType::CUDA:
    case phi::proto::ApiType::CU:
      return "https://docs.nvidia.com/cuda/cuda-runtime-api/"
             "group__CUDART__TYPES.html#group__CUDART__TYPES_"
             "1g3f51e3575c2178246db0a94a430e0038";
      break;
    case phi::proto::ApiType::CURAND:
      return "https://docs.nvidia.com/cuda/curand/"
             "group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437";
      break;
    case phi::proto::ApiType::CUDNN:
      return "https://docs.nvidia.com/deeplearning/cudnn/api/"
             "index.html#cudnnStatus_t";
      break;
    case phi::proto::ApiType::CUBLAS:
      return "https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t";
      break;
    case phi::proto::ApiType::CUSOLVER:
      return "https://docs.nvidia.com/cuda/cusolver/"
             "index.html#cuSolverSPstatus";
      break;
    case phi::proto::ApiType::NCCL:
      return "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/"
             "types.html#ncclresult-t";
      break;
    case phi::proto::ApiType::CUFFT:
      return "https://docs.nvidia.com/cuda/cufft/index.html#cufftresult";
    case phi::proto::ApiType::CUSPARSE:
      return "https://docs.nvidia.com/cuda/cusparse/"
             "index.html#cusparseStatus_t";
      break;
    default:
      return "Unknown type of External API, can't get error message URL!";
      break;
  }
}

template <typename T>
std::string GetExternalErrorMsg(T status) {
  std::ostringstream sout;
  bool _initSucceed = false;
  phi::proto::ExternalErrorDesc externalError;
  if (externalError.ByteSizeLong() == 0) {
    std::string search_path_1;
    std::string search_path_2;
    std::string search_path_3;
#if !defined(_WIN32)
    Dl_info info;
    if (dladdr(
            reinterpret_cast<void*>(common::enforce::GetCurrentTraceBackString),
            &info)) {
      std::string phi_so_path(info.dli_fname);
      const size_t last_slash_idx = phi_so_path.find_last_of('/');
      if (std::string::npos != last_slash_idx) {
        phi_so_path.erase(last_slash_idx, std::string::npos);
      }
      // due to 'phi_so_path' may be 'site-packages/paddle/libs/libphi.so' or
      // 'build/paddle/phi/libphi.so', we have different search path
      search_path_1 =
          phi_so_path +
          "/../include/third_party/externalError/data/externalErrorMsg.pb";
      search_path_2 = phi_so_path +
                      "/../third_party/externalError/data/externalErrorMsg.pb";
      search_path_3 =
          phi_so_path +
          "/../../third_party/externalError/data/externalErrorMsg.pb";
    }
#else
    char buf[512];
    MEMORY_BASIC_INFORMATION mbi;
    HMODULE h_module =
        (::VirtualQuery(common::enforce::GetCurrentTraceBackString,
                        &mbi,
                        sizeof(mbi)) != 0)
            ? (HMODULE)mbi.AllocationBase
            : NULL;
    GetModuleFileName(h_module, buf, 512);
    std::string exe_path(buf);
    const size_t last_slash_idx = exe_path.find_last_of("\\");
    if (std::string::npos != last_slash_idx) {
      exe_path.erase(last_slash_idx, std::string::npos);
    }
    // due to 'exe_path' may be 'site-packages\\paddle\\fluid\\libpaddle.pyd' or
    // 'build\\paddle\\fluid\\platform\\enforce_test.exe', we have different
    // search path
    search_path_1 =
        exe_path +
        "\\..\\include\\third_party\\externalError\\data\\externalErrorMsg.pb";
    search_path_2 =
        exe_path +
        "\\..\\third_party\\externalError\\data\\externalErrorMsg.pb";
    search_path_3 =
        exe_path +
        "\\..\\..\\third_party\\externalError\\data\\externalErrorMsg.pb";
#endif
    std::ifstream fin(search_path_1, std::ios::in | std::ios::binary);
    _initSucceed = externalError.ParseFromIstream(&fin);
    if (!_initSucceed) {
      std::ifstream fin(search_path_2, std::ios::in | std::ios::binary);
      _initSucceed = externalError.ParseFromIstream(&fin);
    }
    if (!_initSucceed) {
      std::ifstream fin(search_path_3, std::ios::in | std::ios::binary);
      _initSucceed = externalError.ParseFromIstream(&fin);
    }
  }
  using __CUDA_STATUS_TYPE__ = decltype(status);
  phi::proto::ApiType proto_type =
      details::ExternalApiProtoType<__CUDA_STATUS_TYPE__>::kProtoType;
  if (_initSucceed) {
    for (int i = 0; i < externalError.errors_size(); ++i) {
      if (proto_type == externalError.errors(i).type()) {
        for (int j = 0; j < externalError.errors(i).messages_size(); ++j) {
          if (status == externalError.errors(i).messages(j).code()) {
            sout << "\n  [Hint: "
                 << externalError.errors(i).messages(j).message() << "]";
            return sout.str();
          }
        }
      }
    }
  }

  sout << "\n  [Hint: Please search for the error code(" << status
       << ") on website (" << GetErrorMsgUrl(status)
       << ") to get Nvidia's official solution and advice about "
       << details::ExternalApiProtoType<__CUDA_STATUS_TYPE__>::kTypeString
       << " Error.]";
  return sout.str();
}

template std::string GetExternalErrorMsg<cudaError_t>(cudaError_t);
template std::string GetExternalErrorMsg<curandStatus_t>(curandStatus_t);
template std::string GetExternalErrorMsg<cudnnStatus_t>(cudnnStatus_t);
template std::string GetExternalErrorMsg<cublasStatus_t>(cublasStatus_t);
template std::string GetExternalErrorMsg<cusparseStatus_t>(cusparseStatus_t);
template std::string GetExternalErrorMsg<cusolverStatus_t>(cusolverStatus_t);
template std::string GetExternalErrorMsg<cufftResult_t>(cufftResult_t);
template std::string GetExternalErrorMsg<CUresult>(CUresult);
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
template std::string GetExternalErrorMsg<ncclResult_t>(ncclResult_t);
#endif

#endif  // PADDLE_WITH_CUDA

}  // namespace phi::enforce
