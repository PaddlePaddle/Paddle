//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <time.h>
#include <fstream>

#include "paddle/fluid/framework/dataset_factory.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace train {

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<paddle::framework::ProgramDesc> LoadProgramDesc(
    const std::string& model_filename) {
    VLOG(3) << "loading model from " << model_filename;
    std::string program_desc_str;
    ReadBinaryFile(model_filename, &program_desc_str);
    std::unique_ptr<paddle::framework::ProgramDesc> main_program(
        new paddle::framework::ProgramDesc(program_desc_str));
    return main_program;
}

}  // namespace train
}  // namespace paddle

int main(int argc, char* argv[]) {
  // filelist, data_feed.prototxt startup_prog, main_prog, model
  std::string filelist = std::string(argv[1]);
  std::vector<std::string> file_vec;
  std::ifstream fin(filelist);
  if (fin) {
    std::string filename;
    while (fin >> filename) {
      file_vec.push_back(filename);
    }
  }

  paddle::framework::InitDevices(false);
  const auto cpu_place = paddle::platform::CPUPlace();
  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  auto startup_program = paddle::train::LoadProgramDesc(std::string(argv[3]));
  auto main_program = paddle::train::LoadProgramDesc(std::string(argv[4]));

  executor.Run(*startup_program, &scope, 0);
  
  std::string data_feed_desc_str;
  paddle::train::ReadBinaryFile(std::string(argv[2]), &data_feed_desc_str);
  VLOG(3) << "load data feed desc done.";
  std::unique_ptr<paddle::framework::Dataset> dataset_ptr;
  dataset_ptr =
      paddle::framework::DatasetFactory::CreateDataset("MultiSlotDataset");
  VLOG(3) << "initialize dataset ptr done";

  int epoch_num = 30;

  std::string loss_name = "mean_0.tmp_0";
  auto loss_var = scope.Var(loss_name);

  for (int epoch = 0; epoch < epoch_num; ++epoch) {
    VLOG(3) << "Epoch:" << epoch;

    // get reader
    dataset_ptr->SetFileList(file_vec);
    VLOG(3) << "set file list done";
    dataset_ptr->SetThreadNum(1);
    VLOG(3) << "set thread num done";
    dataset_ptr->SetDataFeedDesc(data_feed_desc_str);
    VLOG(3) << "set data feed desc done";
    dataset_ptr->CreateReaders();
    const std::vector<paddle::framework::DataFeed*> readers =
      dataset_ptr->GetReaders();
    const std::vector<std::string>& input_feed_names =
      readers[0]->GetUseSlotAlias();
    for (auto name : input_feed_names) {
      readers[0]->AddFeedVar(scope.Var(name), name);
    }
    VLOG(3) << "get reader done";

    readers[0]->Start();
    VLOG(3) << "start a reader";
    PADDLE_ENFORCE_EQ(readers.size(), 1, "readers num should be equal to thread num");
    VLOG(3) << "readers size: " << readers.size();

    int step = 0;
    std::vector<float> loss_vec;
    while (readers[0]->Next() > 0) {
      executor.Run(*main_program, &scope, 0, false, true);
      loss_vec.push_back(loss_var->Get<paddle::framework::LoDTensor>().data<float>()[0]);
    }
    float average_loss = accumulate(loss_vec.begin(), loss_vec.end(), 0.0)/loss_vec.size(); 

    std::cout << "epoch: " << epoch << " average loss: "
              << average_loss
              << std::endl;
  dataset_ptr->DestroyReaders();
  }
}
