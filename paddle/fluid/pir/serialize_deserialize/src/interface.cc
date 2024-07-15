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
#include <stdio.h>
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
  std::ifstream f(file_path);
  Json data = Json::parse(f);
  PatchBuilder builder(pir_version);

  if (data.contains(BASE_CODE) && data[BASE_CODE].contains(MAGIC) &&
      data[BASE_CODE][MAGIC] == PIR) {
    uint64_t file_version =
        data.at(BASE_CODE).at(PIRVERSION).template get<uint64_t>();
    if (file_version != pir_version) {
      std::string cur_file = std::string(__FILE__);
      std::string yaml_file =
          cur_file.substr(0, cur_file.rfind('/')) + "/patch.yaml";
      builder.BuildPatch(yaml_file);  // TODO(czy) : find file patch
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument("Invalid model file."));
  }

  ProgramReader reader(pir_version);
  reader.RecoverProgram(&(data[PROGRAM]), program, &builder);

  if (data[BASE_CODE].contains(TRAINABLE)) {
    return data[BASE_CODE][TRAINABLE].get<bool>();
  } else {
    return false;
  }
}

}  // namespace pir
