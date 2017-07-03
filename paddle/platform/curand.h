#include <curand.h>
#include "paddle/platform/dynamic_loader.h"

namespace paddle {
namespace dyload {
std::once_flag curand_dso_flag;
void *curand_dso_handle = nullptr;
#ifdef PADDLE_USE_DSO
#define DYNAMIC_LOAD_CURAND_WRAP(__name)                                       \
  struct DynLoad__##__name {                                                   \
    template <typename... Args>                                                \
    curandStatus_t operator()(Args... args) {                                  \
      typedef curandStatus_t (*curandFunc)(Args...);                           \
      std::call_once(curand_dso_flag, GetCurandDsoHandle, &curand_dso_handle); \
      void *p_##__name = dlsym(curand_dso_handle, #__name);                    \
      return reinterpret_cast<curandFunc>(p_##__name)(args...);                \
    }                                                                          \
  } __name; /* struct DynLoad__##__name */
#else
#define DYNAMIC_LOAD_CURAND_WRAP(__name)      \
  struct DynLoad__##__name {                  \
    template <typename... Args>               \
    curandStatus_t operator()(Args... args) { \
      return __name(args...);                 \
    }                                         \
  } __name; /* struct DynLoad__##__name */
#endif

/* include all needed curand functions in HPPL */
// clang-format off
#define CURAND_RAND_ROUTINE_EACH(__macro)    \
  __macro(curandCreateGenerator)             \
  __macro(curandSetStream)                   \
  __macro(curandSetPseudoRandomGeneratorSeed)\
  __macro(curandGenerateUniform)             \
  __macro(curandGenerateUniformDouble)       \
  __macro(curandDestroyGenerator)
// clang-format on

CURAND_RAND_ROUTINE_EACH(DYNAMIC_LOAD_CURAND_WRAP)

#undef CURAND_RAND_ROUTINE_EACH
#undef DYNAMIC_LOAD_CURAND_WRAP
}
}  // namespace paddle
