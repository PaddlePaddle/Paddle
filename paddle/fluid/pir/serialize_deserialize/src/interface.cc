// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include <glog/logging.h>
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_deserialize.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_serialize.h"
#include "paddle/phi/common/port.h"

namespace pir {
#define PROGRAM "program"
#define BASE_CODE "base_code"
#define MAGIC "magic"
#define PIRVERSION "version"
#define TRAINABLE "trainable"
#define PIR "pir"
void WriteModule(const pir::Program& program,
                 const std::string& file_path,
                 const uint64_t& pir_version,
                 bool overwrite,
                 bool readable,
                 bool trainable) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      common::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  // write base code
  Json total;

  total[BASE_CODE] = {
      {MAGIC, PIR}, {PIRVERSION, pir_version}, {TRAINABLE, trainable}};

  ProgramWriter writer(pir_version, trainable);
  // write program
  total[PROGRAM] = writer.GetProgramJson(&program);
  std::string total_str;
  if (readable) {
    total_str = total.dump(4);
  } else {
    total_str = total.dump();
  }

  MkDirRecursively(DirName(file_path).c_str());
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", file_path));
  fout << total_str;
  fout.close();
}

bool ReadModule(const std::string& file_path,
                pir::Program* program,
                const uint64_t& pir_version) {
  LOG(INFO) << "Entering ReadModule with file_path" << file_path;
  std::ifstream f(file_path);
  if (!f.is_open()) {
    LOG(ERROR) << "Failed to open file: " << file_path;
    return false;
  }
  LOG(INFO) << "File opened successfully.";
  Json data;
  try {
    data = Json::parse(f);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing JSON: " << e.what();
    return false;
  }

  if (data.contains(BASE_CODE) && data[BASE_CODE].contains(MAGIC) &&
      data[BASE_CODE][MAGIC] == PIR) {
    uint64_t file_version =
        data.at(BASE_CODE).at(PIRVERSION).template get<uint64_t>();
    if (file_version != pir_version) {
      PADDLE_THROW(
          common::errors::InvalidArgument("Invalid model version file."));
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument("Invalid model file."));
  }
  LOG(INFO) << "Model version check passed.";

  ProgramReader reader(pir_version);
  try {
    reader.RecoverProgram(&(data[PROGRAM]), program);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error recovering program: " << e.what();
    return false;
  }

  LOG(INFO) << "Program recovered successfully.";

  if (data[BASE_CODE].contains(TRAINABLE)) {
    return data[BASE_CODE][TRAINABLE].get<bool>();
  } else {
    return false;
  }
}

}  // namespace pir
