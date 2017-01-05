#include <fenv.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "PaddleCAPI.h"
#include "PaddleCAPIPrivate.h"
#include "paddle/trainer/TrainerConfigHelper.h"
#include "paddle/utils/Excepts.h"
#include "paddle/utils/PythonUtil.h"

static void initPaddle(int argc, char** argv) {
  paddle::initMain(argc, argv);
  paddle::initPython(argc, argv);
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
}

extern "C" {
int PDInit(int argc, char** argv) {
  std::vector<char*> realArgv;
  realArgv.reserve(argc + 1);
  realArgv.push_back(strdup(""));
  for (int i = 0; i < argc; ++i) {
    realArgv.push_back(argv[i]);
  }
  initPaddle(argc + 1, realArgv.data());
  free(realArgv[0]);
  return PD_NO_ERROR;
}
}
