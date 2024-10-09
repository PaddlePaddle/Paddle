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
                 uint64_t pir_version,
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
                int64_t pir_version) {
  std::ifstream f(file_path);
  Json data = Json::parse(f);
  if (pir_version < 0) {
    pir_version = DEVELOP_VERSION;
    VLOG(6) << "pir_version is null, get pir_version: " << pir_version;
  }

  PatchBuilder builder(pir_version);

  if (data.contains(BASE_CODE) && data[BASE_CODE].contains(MAGIC) &&
      data[BASE_CODE][MAGIC] == PIR) {
    uint64_t file_version =
        data.at(BASE_CODE).at(PIRVERSION).template get<uint64_t>();
    if (file_version != (uint64_t)pir_version) {
      builder.SetFileVersion(file_version);
      // Set max_version to the max version number of release pir plus 1.
      auto max_version = RELEASE_VERSION + 1;
      // If pir_version_ is not 0, we will build patch from file_version_ to
      // pir_version_; If pir_version_ is 0, we will first build patch from
      // file_version_ to max_version, and then add 0.yaml to the end.
      auto version = pir_version == 0 ? max_version : pir_version;
      VLOG(6) << "file_version: " << file_version
              << ", pir_version: " << pir_version
              << ", final_version: " << version;
      builder.BuildPatch(version, max_version);
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
