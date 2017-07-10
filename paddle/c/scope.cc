#include "scope.h"
#include "private.h"

extern "C" {

paddle_scope_handle paddle_new_scope() {
  auto handle = new paddle::c::ScopeHandle();
  handle->type_ = paddle::c::SCOPE;
  handle->scope_ = std::make_shared<paddle::framework::Scope>();
  return handle;
}

paddle_error PADDLE_API paddle_new_scope_with_parent(
    paddle_scope_handle parent, paddle_scope_handle* new_scope) {
  paddle::c::ScopeHandle* parentPtr;
  auto err = paddle::c::CastHandle(parent, &parentPtr);
  if (err != PADDLE_OK) {
    return err;
  }
  if (new_scope == nullptr) {
    return PADDLE_NULLPTR;
  }
  auto scope_handle = new paddle::c::ScopeHandle();
  scope_handle->scope_ =
      std::make_shared<paddle::framework::Scope>(parentPtr->scope_);
  *new_scope = scope_handle;
  return PADDLE_OK;
}

paddle_error paddle_destroy_scope(paddle_scope_handle hdl) {
  paddle::c::ScopeHandle* ptr;
  auto err = paddle::c::CastHandle(hdl, &ptr);
  if (err == PADDLE_OK) {
    delete ptr;
  }
  return err;
}

paddle_error paddle_scope_get_var(paddle_scope_handle scope, const char* name,
                                  paddle_variable_handle* var) {
  paddle::c::ScopeHandle* ptr;
  auto err = paddle::c::CastHandle(scope, &ptr);
  if (err != PADDLE_OK) {
    return err;
  }
  auto var_ptr = ptr->scope_->GetVariable(name);
  if (var_ptr == nullptr) {
    return PADDLE_OK;
  }
  auto var_handle = new paddle::c::VariableHandle();
  var_handle->var_ = var_ptr;
  *var = var_handle;
  return PADDLE_OK;
}

paddle_error PADDLE_API paddle_scope_create_var(paddle_scope_handle scope,
                                                const char* name,
                                                paddle_variable_handle* var) {
  paddle::c::ScopeHandle* ptr;
  auto err = paddle::c::CastHandle(scope, &ptr);
  if (err != PADDLE_OK) {
    return err;
  }
  auto var_handle = new paddle::c::VariableHandle();
  var_handle->var_ = ptr->scope_->CreateVariable(name);
  *var = var_handle;
  return PADDLE_OK;
}
}