#include <cublas_v2.h>
#include "paddle/platform/dynamic_loader.h"

namespace paddle {
namespace dyload {

std::once_flag cublas_dso_flag;
void *cublas_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DYNAMIC_LOAD_CUBLAS_WRAP(__name)                                       \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    cublasStatus_t operator()(Args... args) {                                  \
      typedef cublasStatus_t (*cublasFunc)(Args...);                           \
      std::call_once(cublas_dso_flag, GetCublasDsoHandle, &cublas_dso_handle); \
      void *p_##__name = dlsym(cublas_dso_handle, #__name);                    \
      return reinterpret_cast<cublasFunc>(p_##__name)(args...);                \
    }                                                                          \
  } __name;  // struct DynLoad__##__name
#else
#define DYNAMIC_LOAD_CUBLAS_WRAP(__name)      \
  struct DynLoad__##__name {                  \
    template <typename... Args>               \
    cublasStatus_t operator()(Args... args) { \
      return __name(args...);                 \
    }                                         \
  } __name;  // struct DynLoad__##__name
#endif

#define DYNAMIC_LOAD_CUBLAS_V2_WRAP(__name) DYNAMIC_LOAD_CUBLAS_WRAP(__name)

// include all needed cublas functions in HPPL
// clang-format off
#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSgemv)                    \
  __macro(cublasDgemv)                    \
  __macro(cublasSgemm)                    \
  __macro(cublasDgemm)                    \
  __macro(cublasSgeam)                    \
  __macro(cublasDgeam)                    \

DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasCreate)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasDestroy)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasSetStream)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasSetPointerMode)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasGetPointerMode)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasCgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasZgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgetrfBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgetriBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgetrfBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgetriBatched)
CUBLAS_BLAS_ROUTINE_EACH(DYNAMIC_LOAD_CUBLAS_V2_WRAP)

#undef DYNAMIC_LOAD_CUBLAS_WRAP
#undef DYNAMIC_LOAD_CUBLAS_V2_WRAP
#undef CUBLAS_BLAS_ROUTINE_EACH

// clang-format on
#ifndef PADDLE_TYPE_DOUBLE
#define CUBLAS_GEAM dynload::cublasSgeam
#define CUBLAS_GEMV dynload::cublasSgemv
#define CUBLAS_GEMM dynload::cublasSgemm
#define CUBLAS_GETRF dynload::cublasSgetrfBatched
#define CUBLAS_GETRI dynload::cublasSgetriBatched
#else
#define CUBLAS_GEAM dynload::cublasDgeam
#define CUBLAS_GEMV dynload::cublasDgemv
#define CUBLAS_GEMM dynload::cublasDgemm
#define CUBLAS_GETRF dynload::cublasDgetrfBatched
#define CUBLAS_GETRI dynload::cublasDgetriBatched
#endif
}  // namespace dyload
}  // namespace paddle
