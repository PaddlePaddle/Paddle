#include "mlu_info.h"

#include <glog/logging.h>

void MLUGetDriverVersion(int* x, int* y, int* z) {
  auto status = cnGetDriverVersion(x, y, z);
  LOG(INFO) << "Status: " << status << " MLU driver version: " << *x << "." << *y << "." << *z;
}

int main() {
  int x,y,z;
  MLUGetDriverVersion(&x, &y, &z);
  LOG(INFO) << x << "." << y << "." << z;
  return 0;
}