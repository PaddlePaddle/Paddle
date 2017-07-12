#include <paddle/platform/dynload/curand.h>

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag curand_dso_flag;
void *curand_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CURAND_RAND_ROUTINE_EACH(DEFINE_WRAP);
}
}
}