#include "paddle/fluid/platform/mlu/mlu_info.h"

#include <glog/logging.h>

void GetMLUDriverVersion(int* x, int* y, int* z) {
  if (cnInit(0) == CN_SUCCESS) {
    LOG(INFO) << "Init mlu device success! ";
  } else {
    LOG(INFO) << "Init mlu device failed! ";
  }
  
  auto status = cnGetDriverVersion(x, y, z);
  LOG(INFO) << "Status: " << status << " MLU driver version: " << *x << "." << *y << "." << *z;
}

int main() {
  int x,y,z;
  GetMLUDriverVersion(&x, &y, &z);
  return 0;
}
