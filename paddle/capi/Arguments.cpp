#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"

#define cast(v) paddle::capi::cast<paddle::capi::CArguments>(v)

extern "C" {
int PDArgsCreateNone(PD_Arguments* args) {
  auto ptr = new paddle::capi::CArguments();
  *args = ptr;
  return PD_NO_ERROR;
}

int PDArgsDestroy(PD_Arguments args) {
  if (args == nullptr) return PD_NULLPTR;
  delete cast(args);
  return PD_NO_ERROR;
}

int PDArgsGetSize(PD_Arguments args, uint64_t* size) {
  if (args == nullptr || size == nullptr) return PD_NULLPTR;
  *size = cast(args)->args.size();
  return PD_NO_ERROR;
}

int PDArgsResize(PD_Arguments args, uint64_t size) {
  if (args == nullptr) return PD_NULLPTR;
  cast(args)->args.resize(size);
  return PD_NO_ERROR;
}

int PDArgsSetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat) {
  if (args == nullptr || mat == nullptr) return PD_NULLPTR;
  auto m = paddle::capi::cast<paddle::capi::CMatrix>(mat);
  if (m->mat == nullptr) return PD_NULLPTR;
  auto a = cast(args);
  if (ID >= a->args.size()) return PD_OUT_OF_RANGE;
  a->args[ID].value = m->mat;
  return PD_NO_ERROR;
}

int PDArgsGetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat) {
  if (args == nullptr || mat == nullptr) return PD_NULLPTR;
  auto m = paddle::capi::cast<paddle::capi::CMatrix>(mat);
  auto a = cast(args);
  if (ID >= a->args.size()) return PD_OUT_OF_RANGE;
  m->mat = a->args[ID].value;
  return PD_NO_ERROR;
}
}
