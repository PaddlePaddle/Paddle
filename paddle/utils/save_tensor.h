/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <iostream>

#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/version.h"
#include "glog/logging.h"

namespace paddle {

// void SerializeToStream(std::ostream &os,
//                        const phi::DenseTensor &tensor,
//                        const platform::DeviceContext &dev_ctx) {
//   {  // the 1st field, uint32_t version for DenseTensor
//     os.write(
//         reinterpret_cast<const char
//         *>(&paddle::framework::kCurTensorVersion),
//         sizeof(paddle::framework::kCurTensorVersion));
//   }
//   {
//     // the 2st field, LoD information
//     // uint64_t lod_level
//     // uint64_t lod_level_1 size in byte.
//     // int*     lod_level_1 data
//     // ...
//     auto lod = tensor.lod();
//     uint64_t size = lod.size();
//     os.write(reinterpret_cast<const char *>(&size), sizeof(size));

//     for (auto &each : lod) {
//       size = each.size() * sizeof(framework::LoD::value_type::value_type);
//       os.write(reinterpret_cast<const char *>(&size), sizeof(size));
//       os.write(reinterpret_cast<const char *>(each.data()),
//                static_cast<std::streamsize>(size));
//     }
//   }
//   // the 3st field, Tensor
//   paddle::framework::TensorToStream(
//       os, static_cast<phi::DenseTensor>(tensor), dev_ctx);
// }

// void SerializeToStream(std::ostream &os, const phi::DenseTensor &tensor) {
//   platform::DeviceContextPool &pool =
//   platform::DeviceContextPool::Instance(); const platform::DeviceContext
//   *dev_ctx; auto place = tensor.place(); dev_ctx = pool.Get(place);
//   SerializeToStream(os, tensor, *dev_ctx);
// }

// void DeserializeFromStream(std::istream &is,
//                            phi::DenseTensor *tensor,
//                            const platform::DeviceContext &dev_ctx,
//                            const size_t &seek,
//                            const std::vector<int64_t> &shape) {
//   {
//     // the 1st field, unit32_t version for DenseTensor
//     uint32_t version;
//     is.read(reinterpret_cast<char *>(&version), sizeof(version));
//     PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
//                       true,
//                       phi::errors::InvalidArgument(
//                           "Tensor version %u is not supported.", version));
//     PADDLE_ENFORCE_EQ(
//         version,
//         0U,
//         phi::errors::InvalidArgument(
//             "Deserialize to tensor failed, maybe the loaded file is "
//             "not a paddle model(expected file format: 0, but %u found).",
//             version));
//   }
//   {
//     // the 2st field, LoD information
//     uint64_t lod_level;
//     is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
//     auto &lod = *tensor->mutable_lod();
//     lod.resize(lod_level);
//   }
//   // the 3st filed, Tensor
//   paddle::framework::TensorFromStream(
//       is, static_cast<phi::DenseTensor *>(tensor), dev_ctx, seek, shape);
// }

// void DeserializeFromStream(std::istream &is,
//                            phi::DenseTensor *tensor,
//                            const platform::DeviceContext &dev_ctx) {
//   {
//     // the 1st field, unit32_t version for DenseTensor
//     uint32_t version;
//     is.read(reinterpret_cast<char *>(&version), sizeof(version));
//     PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
//                       true,
//                       phi::errors::InvalidArgument(
//                           "Tensor version %u is not supported.", version));
//     PADDLE_ENFORCE_EQ(
//         version,
//         0U,
//         phi::errors::InvalidArgument(
//             "Deserialize to tensor failed, maybe the loaded file is "
//             "not a paddle model(expected file format: 0, but %u found).",
//             version));
//   }
//   {
//     // the 2st field, LoD information
//     uint64_t lod_level;
//     is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
//     auto &lod = *tensor->mutable_lod();
//     lod.resize(lod_level);
//     for (uint64_t i = 0; i < lod_level; ++i) {
//       uint64_t size;
//       is.read(reinterpret_cast<char *>(&size), sizeof(size));
//       std::vector<size_t> tmp(size / sizeof(size_t));
//       is.read(reinterpret_cast<char *>(tmp.data()),
//               static_cast<std::streamsize>(size));
//       lod[i] = tmp;
//     }
//   }
//   // the 3st filed, Tensor
//   paddle::framework::TensorFromStream(
//       is, static_cast<phi::DenseTensor *>(tensor), dev_ctx);
// }

// void DeserializeFromStream(std::istream &os, phi::DenseTensor *tensor) {
//   platform::DeviceContextPool &pool =
//   platform::DeviceContextPool::Instance(); const platform::DeviceContext
//   *dev_ctx; dev_ctx = pool.Get(platform::CPUPlace());
//   DeserializeFromStream(os, tensor, *dev_ctx);
// }

void SaveTensor(
                const phi::DenseTensor& x,
                const std::string& file_path,
                bool overwrite,
                bool save_as_fp16) {
    std::string new_path(file_path);
    std::cout<<"new path : "<<new_path<<std::endl;
    if (FileExists(new_path)){
     std::cout<<"FileExists : "<<new_path<<"pass"<<std::endl;
     return ;
    }
    // while (FileExists(new_path)){
    // auto pos = new_path.find_last_of('_');
    // if (pos == new_path.npos){
    //     return ;
    // }
    // auto pre = new_path.substr(0, pos+1);
    // auto num = std::atoi(new_path.substr(pos + 1).c_str());
    // new_path = pre + std::to_string(num+1);
    // }
//   PADDLE_ENFORCE_EQ(
//       FileExists(file_path) && !overwrite,
//       false,
//       phi::errors::PreconditionNotMet(
//           "%s exists!, cannot save to it when overwrite is set to false.",
//           file_path,
//           overwrite));

  VLOG(6)<<"SAVE TO "<<new_path;
  MkDirRecursively(DirName(new_path).c_str());

  std::ofstream fout(new_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fout),
      true,
      phi::errors::Unavailable("Cannot open %s to save variables.", new_path));
  VLOG(6)<<"START SerializeToStream";
  framework::SerializeToStream(fout, x);
  VLOG(6)<<"end SerializeToStream";

  fout.close();
}

}  // namespace paddle
