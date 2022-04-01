// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace data {
using LoDTensor = framework::LoDTensor;
using LoDTensorArray = framework::LoDTensorArray;

#ifdef _WIN32
constexpr char DIR_SEP = '\\';
#else
constexpr char DIR_SEP = '/';
#endif

static std::string JoinPath(const std::string path1, const std::string path2) {
  // empty check
  if (path1.empty()) return path2;
  if (path1.empty()) return path1;

  // absolute path check
  if (path2[0] == DIR_SEP) return path2;
#ifdef _WIN32
  if (path2[1] == ":") return path2;
#endif

  // concat path
  if (path1[path1.length() - 1] == DIR_SEP) return path1 + path2;
  return path1 + DIR_SEP + path2;
}

static void ParseFilesAndLabels(
    const std::string data_root,
    std::vector<std::pair<std::string, int>>* samples) {
  auto* dir = opendir(data_root.c_str());
  PADDLE_ENFORCE_NE(dir, nullptr, platform::errors::InvalidArgument(
                                      "Cannot open directory %s", data_root));

  // Step 1: parse classes info
  std::vector<std::string> classes;
  auto* entry = readdir(dir);
  while (entry) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      entry = readdir(dir);
      continue;
    }

    auto cls_path = JoinPath(data_root, entry->d_name);
    struct stat s;
    int ret = stat(cls_path.c_str(), &s);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::InvalidArgument(
                                  "Directory %s is unaccessiable.", cls_path));

    if (S_ISDIR(s.st_mode)) classes.emplace_back(entry->d_name);

    entry = readdir(dir);
  }

  closedir(dir);

  // sort directories in alphabetic order to generate class order
  std::sort(classes.begin(), classes.end());

  // Step 2: traverse directory to generate samples
  for (int class_id = 0; class_id < static_cast<int>(classes.size());
       class_id++) {
    auto cur_dir = data_root + DIR_SEP + classes[class_id];
    dir = opendir(cur_dir.c_str());
    entry = readdir(dir);
    while (entry) {
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
        entry = readdir(dir);
        continue;
      }

      auto file = cur_dir + DIR_SEP + entry->d_name;
      samples->emplace_back(std::make_pair(file, class_id));

      entry = readdir(dir);
    }
    closedir(dir);
  }
}

std::map<std::string, std::vector<std::pair<std::string, int>>>
    root_to_samples_;

static std::vector<std::pair<std::string, int>>* GetFilesAndLabelsFromCache(
    const std::string data_root) {
  auto iter = root_to_samples_.find(data_root);
  if (iter == root_to_samples_.end()) {
    std::vector<std::pair<std::string, int>> samples;
    ParseFilesAndLabels(data_root, &samples);
    VLOG(4) << "Init sample number: " << samples.size();
    root_to_samples_[data_root] = samples;
  }

  return &(root_to_samples_[data_root]);
}

template <typename T>
class FileLabelLoaderCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* indices = ctx.Input<LoDTensor>("Indices");
    auto image_arr = ctx.MultiOutput<LoDTensor>("Image");
    auto* label_tensor = ctx.Output<LoDTensor>("Label");

    auto data_root = ctx.Attr<std::string>("data_root");
    auto* samples = GetFilesAndLabelsFromCache(data_root);

    auto batch_size = indices->dims()[0];
    const int64_t* indices_data = indices->data<int64_t>();

    label_tensor->Resize(phi::make_ddim({static_cast<int64_t>(batch_size)}));
    auto* label_data =
        label_tensor->mutable_data<int64_t>(platform::CPUPlace());
    for (int64_t i = 0; i < batch_size; i++) {
      int64_t index = static_cast<int>(indices_data[i]);
      auto file = samples->at(index).first;
      auto label = samples->at(index).second;
      std::ifstream input(file.c_str(),
                          std::ios::in | std::ios::binary | std::ios::ate);
      std::streamsize file_size = input.tellg();

      input.seekg(0, std::ios::beg);

      auto image = image_arr[i];
      std::vector<int64_t> image_len = {file_size};
      image->Resize(phi::make_ddim(image_len));

      uint8_t* data = image->mutable_data<uint8_t>(platform::CPUPlace());

      input.read(reinterpret_cast<char*>(data), file_size);

      label_data[i] = static_cast<int64_t>(label);
    }
  }

 private:
  void copy_tensor(const framework::LoDTensor& lod_tensor,
                   framework::LoDTensor* out) const {
    if (lod_tensor.numel() == 0) return;
    auto& out_tensor = *out;
    framework::TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle
