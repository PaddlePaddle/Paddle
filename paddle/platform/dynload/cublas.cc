#include <paddle/platform/dynload/cublas.h>

namespace paddle {
namespace platform {
namespace dynload {
std::once_flag cublas_dso_flag;
void *cublas_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUBLAS_BLAS_ROUTINE_EACH(DEFINE_WRAP);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
