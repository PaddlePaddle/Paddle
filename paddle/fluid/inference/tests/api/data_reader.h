// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <string>

namespace paddle {

// Data reader for imagenet for ResNet50
struct DataReader {
  static void drawImages(float* input, bool is_rgb, int batch_size,
                         int channels, int width, int height);

  explicit DataReader(const std::string& data_list_path,
                      const std::string& data_dir_path, int width, int height,
                      int channels, bool convert_to_rgb);

  // return true if separator works or false otherwise
  bool SetSeparator(char separator);

  bool NextBatch(float* input, int64_t* label, int batch_size,
                 bool debug_display_images);

 private:
  std::string data_list_path;
  std::string data_dir_path;
  std::ifstream file;
  int width;
  int height;
  int channels;
  bool convert_to_rgb;
  char sep{'\t'};
};

}  // namespace paddle
