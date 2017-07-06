#pragma once

namespace paddle {
namespace c {

enum ObjectType {
  SCOPE = 0,
};

#define PADDLE_HANDLE_HEADER ::paddle::c::ObjectType type_;

struct HandleBase {
  PADDLE_HANDLE_HEADER
};
}
}
