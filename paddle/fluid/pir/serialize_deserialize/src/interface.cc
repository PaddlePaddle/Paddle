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
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_deserialize.h"
#include "paddle/fluid/pir/serialize_deserialize/include/ir_serialize.h"
#include "paddle/phi/backends/dynload/port.h"

namespace pir {

void WriteModule(const pir::Program& program,
                 const std::string& file_path,
                 const uint64_t& pir_version,
                 bool overwrite,
                 bool readable) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      common::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  // write base code
  Json total;

  total["base_code"] = {{"magic", "PIR"}, {"version", pir_version}};

  auto t1 = std::chrono::high_resolution_clock::now();
  ProgramWriter writer(pir_version);
  // write program
  total["program"] = writer.GetProgramJson(&program);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::string total_str;
  if (readable) {
    total_str = total.dump(4);
  } else {
    total_str = total.dump();
  }

  MkDirRecursively(DirName(file_path).c_str());
  auto t3 = std::chrono::high_resolution_clock::now();
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", file_path));
  fout << total_str;
  fout.close();
  auto t4 = std::chrono::high_resolution_clock::now();

  auto time_1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  auto time_2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
  auto time_3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3);
  // 输出时间差
  std::cout << "serialize time: " << time_1.count() << " microseconds"
            << std::endl;
  std::cout << "dump time: " << time_2.count() << " microseconds" << std::endl;
  std::cout << "file write time: " << time_3.count() << " microseconds"
            << std::endl;
}

void ReadModule(const std::string& file_path, pir::Program* program) {
  std::ifstream f(file_path);
  Json data = Json::parse(f);
}

}  // namespace pir
