/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <fstream>
#include <iostream>
#include "TrainerConfig.pb.h"

bool loadPb(google::protobuf::Message* conf, const std::string& filename) {
  std::ifstream fin;
  fin.open(filename.c_str());
  if (fin.is_open()) {
    std::string str((std::istreambuf_iterator<char>(fin)),
                    std::istreambuf_iterator<char>());
    bool ok = google::protobuf::TextFormat::ParseFromString(str, conf);
    fin.close();
    return ok;
  } else {
    return false;
  }
}

int main(int argc, char** argv) {
  std::unique_ptr<google::protobuf::Message> config1;
  std::unique_ptr<google::protobuf::Message> config2;
  if (argc == 3) {
    config1.reset(new paddle::ModelConfig());
    config2.reset(new paddle::ModelConfig());
  } else if (argc == 4) {
    config1.reset(new paddle::TrainerConfig());
    config2.reset(new paddle::TrainerConfig());
  }
  if (!config1 || !config2) {
    return 1;
  } else if (!loadPb(config1.get(), argv[1])) {
    return 2;
  } else if (!loadPb(config2.get(), argv[2])) {
    return 3;
  } else {
    if (google::protobuf::util::MessageDifferencer::ApproximatelyEquals(
            *config1, *config2)) {
      return 0;
    } else {
      return 4;
    }
  }
}
