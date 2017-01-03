#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"

#define cast(v) paddle::capi::cast<paddle::capi::CVector>(v)
extern "C" {
int PDVecCreate(PD_Vector* vec, uint64_t size, bool useGpu) {
  auto ptr = new paddle::capi::CVector();
  ptr->vec = paddle::Vector::create(size, useGpu);
  *vec = ptr;
  return PD_NO_ERROR;
}
int PDVecDestroy(PD_Vector vec) {
  auto v = cast(vec);
  v->vec.reset();
  delete v;
  return PD_NO_ERROR;
}

int PDVecIsSparse(PD_Vector vec, bool* isSparse) {
  if (isSparse == nullptr || vec == nullptr) {
    return PD_NULLPTR;
  }
  *isSparse = cast(vec)->vec->isSparse();
  return PD_NO_ERROR;
}
}
