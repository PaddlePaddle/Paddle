#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"

using paddle::capi::cast;

extern "C" {

int PDIVecCreateNone(PD_IVector* ivec) {
  if (ivec == nullptr) return kPD_NULLPTR;
  auto ptr = new paddle::capi::CIVector();
  *ivec = ptr;
  return kPD_NO_ERROR;
}

int PDIVecDestroy(PD_IVector ivec) {
  if (ivec == nullptr) return kPD_NULLPTR;
  delete cast<paddle::capi::CIVector>(ivec);
  return kPD_NO_ERROR;
}

int PDIVectorGet(PD_IVector ivec, int** buffer) {
  if (ivec == nullptr || buffer == nullptr) return kPD_NULLPTR;
  auto v = cast<paddle::capi::CIVector>(ivec);
  if (v->vec == nullptr) return kPD_NULLPTR;
  *buffer = v->vec->getData();
  return kPD_NO_ERROR;
}
}
