#include "scope.h"
#include <paddle/framework/scope.h>
#include "private.h"
namespace paddle {
namespace c {
struct ScopeHandle {
  PADDLE_HANDLE_HEADER
  std::shared_ptr<paddle::framework::Scope> scope_;
};
}  // namespace c
}  // namespace paddle

extern "C" {

paddle_scope_handle paddle_new_scope() {
  auto handle = new paddle::c::ScopeHandle();
  handle->type_ = paddle::c::SCOPE;
  handle->scope_ = std::make_shared<paddle::framework::Scope>();
  return handle;
}

paddle_error paddle_destroy_scope(paddle_scope_handle hdl) {
  if (reinterpret_cast<paddle::c::HandleBase*>(hdl)->type_ !=
      paddle::c::SCOPE) {
    return PADDLE_TYPE_MISMATCH;
  } else {
    auto handle = reinterpret_cast<paddle::c::ScopeHandle*>(hdl);
    delete handle;
    return PADDLE_OK;
  }
}
}