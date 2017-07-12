#include <paddle/platform/device_context.h>

namespace paddle {
namespace platform {
namespace dynload {
namespace dummy {
// Make DeviceContext A library.
int DUMMY_VAR_FOR_DEV_CTX = 0;

}  // namespace dummy
}  // namespace dynload
}  // namespace platform
}  // namespace paddle