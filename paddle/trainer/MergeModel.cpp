/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>

#include "ParamUtil.h"
#include "Trainer.h"
#include "paddle/pserver/ParameterServer2.h"
#include "paddle/utils/PythonUtil.h"

DEFINE_string(model_dir, "", "Directory for separated model files");
DEFINE_string(config_file, "", "Config file for the model");
DEFINE_string(model_file, "", "File for merged model file");

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

int main(int argc, char** argv) {
  initMain(argc, argv);
  initPython(argc, argv);

  if (FLAGS_model_dir.empty() || FLAGS_config_file.empty() ||
      FLAGS_model_file.empty()) {
    LOG(INFO) << "Usage: ./paddle_merge_model --model_dir=pass-00000 "
                 "--config_file=config.py --model_file=out.paddle";
    return 0;
  }

  string confFile = FLAGS_config_file;
#ifndef PADDLE_WITH_CUDA
  FLAGS_use_gpu = false;
#endif
  auto config = std::make_shared<TrainerConfigHelper>(confFile);
  unique_ptr<GradientMachine> gradientMachine(GradientMachine::create(*config));
  gradientMachine->loadParameters(FLAGS_model_dir);

  ofstream os(FLAGS_model_file);

  string buf;
  config->getConfig().SerializeToString(&buf);
  int64_t size = buf.size();
  os.write((char*)&size, sizeof(size));
  CHECK(os) << "Fail to write to " << FLAGS_model_file;
  os.write(buf.data(), buf.size());
  vector<ParameterPtr>& parameters = gradientMachine->getParameters();
  for (auto& para : parameters) {
    para->save(os);
    CHECK(os) << "Fail to write to " << FLAGS_model_file;
  }
  os.close();

  return 0;
}
