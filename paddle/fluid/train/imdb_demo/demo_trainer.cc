//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "include/save_model.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/dataset_factory.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

#include "gflags/gflags.h"

DEFINE_string(filelist, "train_filelist.txt", "filelist for fluid dataset");
DEFINE_string(data_proto_desc, "data.proto", "data feed protobuf description");
DEFINE_string(startup_program_file, "startup_program",
              "startup program description");
DEFINE_string(main_program_file, "", "main program description");
DEFINE_string(loss_name, "mean_0.tmp_0",
              "loss tensor name in the main program");
DEFINE_string(save_dir, "cnn_model", "directory to save trained models");
DEFINE_int32(epoch_num, 30, "number of epochs to run when training");

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

bool IsPersistable(const paddle::framework::VarDesc* var) {
  if (var->Persistable() &&
      var->GetType() != paddle::framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != paddle::framework::proto::VarType::FETCH_LIST &&
      var->GetType() != paddle::framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}

}  // namespace train
}  // namespace paddle

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cerr << "filelist: " << FLAGS_filelist << std::endl;
  std::cerr << "data_proto_desc: " << FLAGS_data_proto_desc << std::endl;
  std::cerr << "startup_program_file: " << FLAGS_startup_program_file
            << std::endl;
  std::cerr << "main_program_file: " << FLAGS_main_program_file << std::endl;
  std::cerr << "loss_name: " << FLAGS_loss_name << std::endl;
  std::cerr << "save_dir: " << FLAGS_save_dir << std::endl;
  std::cerr << "epoch_num: " << FLAGS_epoch_num << std::endl;

  std::string filelist = std::string(FLAGS_filelist);
  std::vector<std::string> file_vec;
  std::ifstream fin(filelist);
  if (fin) {
    std::string filename;
    while (fin >> filename) {
      file_vec.push_back(filename);
    }
  }
  PADDLE_ENFORCE_GE(file_vec.size(), 1, "At least one file to train");
  paddle::framework::InitDevices(false);
  const auto cpu_place = paddle::platform::CPUPlace();
  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  auto startup_program =
      paddle::train::LoadProgramDesc(std::string(FLAGS_startup_program_file));
  auto main_program =
      paddle::train::LoadProgramDesc(std::string(FLAGS_main_program_file));

  executor.Run(*startup_program, &scope, 0);

  std::string data_feed_desc_str;
  paddle::train::ReadBinaryFile(std::string(FLAGS_data_proto_desc),
                                &data_feed_desc_str);
  VLOG(3) << "load data feed desc done.";
  std::unique_ptr<paddle::framework::Dataset> dataset_ptr;
  dataset_ptr =
      paddle::framework::DatasetFactory::CreateDataset("MultiSlotDataset");
  VLOG(3) << "initialize dataset ptr done";

  // find all params
  std::vector<std::string> param_names;
  const paddle::framework::BlockDesc& global_block = main_program->Block(0);
  for (auto* var : global_block.AllVars()) {
    if (paddle::train::IsPersistable(var)) {
      VLOG(3) << "persistable variable's name: " << var->Name();
      param_names.push_back(var->Name());
    }
  }

  int epoch_num = FLAGS_epoch_num;
  std::string loss_name = FLAGS_loss_name;
  auto loss_var = scope.Var(loss_name);

  LOG(INFO) << "Start training...";

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
    PADDLE_ENFORCE_EQ(readers.size(), 1,
                      "readers num should be equal to thread num");
    readers[0]->SetPlace(paddle::platform::CPUPlace());
    const std::vector<std::string>& input_feed_names =
        readers[0]->GetUseSlotAlias();
    for (auto name : input_feed_names) {
      readers[0]->AddFeedVar(scope.Var(name), name);
    }
    VLOG(3) << "get reader done";
    readers[0]->Start();
    VLOG(3) << "start a reader";
    VLOG(3) << "readers size: " << readers.size();

    int step = 0;
    std::vector<float> loss_vec;

    while (readers[0]->Next() > 0) {
      executor.Run(*main_program, &scope, 0, false, true);
      loss_vec.push_back(
          loss_var->Get<paddle::framework::LoDTensor>().data<float>()[0]);
    }
    float average_loss =
        accumulate(loss_vec.begin(), loss_vec.end(), 0.0) / loss_vec.size();

    LOG(INFO) << "epoch: " << epoch << "; average loss: " << average_loss;
    dataset_ptr->DestroyReaders();

    // save model
    std::string save_dir_root = FLAGS_save_dir;
    std::string save_dir =
        save_dir_root + "/epoch" + std::to_string(epoch) + ".model";
    paddle::framework::save_model(main_program, &scope, param_names, save_dir,
                                  false);
  }
}
