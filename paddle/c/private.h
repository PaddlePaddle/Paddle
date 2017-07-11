#pragma once
#include <paddle/framework/scope.h>
#include <paddle/framework/variable.h>
#include "error.h"

namespace paddle {
namespace c {

enum ObjectType { SCOPE = 0, VARIABLE = 1 };

#define PADDLE_HANDLE_HEADER ::paddle::c::ObjectType type_;

#define PADDLE_HANDLE_TYPE(type)       \
  ::paddle::c::ObjectType type_{type}; \
  constexpr static ObjectType TYPE = type;

struct HandleBase {
  PADDLE_HANDLE_HEADER
};

struct VariableHandle {
  PADDLE_HANDLE_TYPE(VARIABLE);
  paddle::framework::Variable* var_{nullptr};
};

struct ScopeHandle {
  PADDLE_HANDLE_TYPE(SCOPE);
  std::shared_ptr<paddle::framework::Scope> scope_;
};

template <typename T>
static inline paddle_error CastHandle(void* hdl, T** ptr) {
  if (hdl == nullptr) {
    return PADDLE_NULLPTR;
  } else if (reinterpret_cast<T*>(hdl)->type_ != T::TYPE) {
    return PADDLE_TYPE_MISMATCH;
  } else {
    *ptr = reinterpret_cast<T*>(hdl);
    return PADDLE_OK;
  }
}

}  // namespace c
}  // namespace paddle
