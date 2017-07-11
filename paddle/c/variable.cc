#include "variable.h"
#include "private.h"

extern "C" {

paddle_error paddle_destroy_variable(paddle_variable_handle var_handle) {
  paddle::c::VariableHandle* var;
  auto err = paddle::c::CastHandle(var_handle, &var);
  if (err == PADDLE_OK) {
    delete var;
  }
  return err;
}
}