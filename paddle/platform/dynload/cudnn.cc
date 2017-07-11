#include <paddle/platform/dynload/cudnn.h>

namespace paddle {
namespace platform {
namespace dynload {
std::once_flag cudnn_dso_flag;
void* cudnn_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUDNN_DNN_ROUTINE_EACH(DEFINE_WRAP);
CUDNN_DNN_ROUTINE_EACH_R2(DEFINE_WRAP);

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_R3
CUDNN_DNN_ROUTINE_EACH_AFTER_R3(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_R4
CUDNN_DNN_ROUTINE_EACH_AFTER_R4(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_R5
CUDNN_DNN_ROUTINE_EACH_R5(DEFINE_WRAP);
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle