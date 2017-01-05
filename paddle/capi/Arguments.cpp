#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"

using paddle::capi::cast;

#define castArg(v) cast<paddle::capi::CArguments>(v)
#define castIVec(v) cast<paddle::capi::CIVector>(v)

extern "C" {
int PDArgsCreateNone(PD_Arguments* args) {
  auto ptr = new paddle::capi::CArguments();
  *args = ptr;
  return kPD_NO_ERROR;
}

int PDArgsDestroy(PD_Arguments args) {
  if (args == nullptr) return kPD_NULLPTR;
  delete castArg(args);
  return kPD_NO_ERROR;
}

int PDArgsGetSize(PD_Arguments args, uint64_t* size) {
  if (args == nullptr || size == nullptr) return kPD_NULLPTR;
  *size = castArg(args)->args.size();
  return kPD_NO_ERROR;
}

int PDArgsResize(PD_Arguments args, uint64_t size) {
  if (args == nullptr) return kPD_NULLPTR;
  castArg(args)->args.resize(size);
  return kPD_NO_ERROR;
}

int PDArgsSetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat) {
  if (args == nullptr || mat == nullptr) return kPD_NULLPTR;
  auto m = paddle::capi::cast<paddle::capi::CMatrix>(mat);
  if (m->mat == nullptr) return kPD_NULLPTR;
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  a->args[ID].value = m->mat;
  return kPD_NO_ERROR;
}

int PDArgsGetValue(PD_Arguments args, uint64_t ID, PD_Matrix mat) {
  if (args == nullptr || mat == nullptr) return kPD_NULLPTR;
  auto m = paddle::capi::cast<paddle::capi::CMatrix>(mat);
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  m->mat = a->args[ID].value;
  return kPD_NO_ERROR;
}

int PDArgsGetIds(PD_Arguments args, uint64_t ID, PD_IVector ids) {
  if (args == nullptr || ids == nullptr) return kPD_NULLPTR;
  auto iv = castIVec(ids);
  auto a = castArg(args);
  if (ID >= a->args.size()) return kPD_OUT_OF_RANGE;
  iv->vec = a->args[ID].ids;
  return kPD_NO_ERROR;
}
}
