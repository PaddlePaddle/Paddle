// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

std::vector<std::string> split(const std::string& str, const char* splitter) {
  std::vector<std::string> results;
  std::string::size_type pos1, pos2;
  pos2 = str.find(splitter);
  pos1 = 0;
  while (std::string::npos != pos2) {
    results.push_back(str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = str.find(splitter, pos1);
  }
  if (pos1 != str.length()) {
    results.push_back(str.substr(pos1));
  }
  return results;
}

struct InputMeta {
  std::string name;
  std::string alias;
  paddle_infer::DataType dtype;
  std::vector<int> shape;
};

struct LoadConfig {
  std::string file_path;
  size_t capability{0};
  std::vector<InputMeta> ids;
  std::vector<InputMeta> others;
};

class RuntimeContext {
 public:
  explicit RuntimeContext(LoadConfig config) : config_{std::move(config)} {}
  const LoadConfig& config() const { return config_; }
  const std::fstream& f() const { return f_; }

 private:
  LoadConfig config_;
  std::fstream f_;
};

struct EOFException : std::exception {
  const char* what() const noexcept override { return "EOFException\n"; }
};

struct TensorHandle {
 public:
  TensorHandle(const InputMeta& meta,
               const std::vector<std::string>& str,
               const char* splitter) {
    if (meta.dtype == paddle_infer::DataType::FLOAT32) {
      paddle::PaddleBuf buf(str.size() * sizeof(float));
      auto* ptr = reinterpret_cast<float*>(buf.data());
      for (const auto& s : str) {
        *ptr++ = std::stof(s);
      }
      tensor_.data = std::move(buf);
    } else if (meta.dtype == paddle_infer::DataType::INT64) {
      paddle::PaddleBuf buf(str.size() * sizeof(int64_t));
      auto* ptr = reinterpret_cast<int64_t*>(buf.data());
      for (const auto& s : str) {
        *ptr++ = std::stoi(s);
      }
      tensor_.data = std::move(buf);
    } else {
      LOG(FATAL) << "unsupported data type!";
    }
    tensor_.shape = {1, str.size(), 1};
    tensor_.name = meta.name;
    tensor_.dtype = meta.dtype;
  }
  paddle::PaddleTensor release() { return std::move(tensor_); }

 private:
  paddle::PaddleTensor tensor_;
};

class FeedData {
 public:
  void push(std::vector<TensorHandle> value) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cv_.notify_one();
  }

  std::vector<TensorHandle> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty() && !finished_) {
      cv_.wait(lock, [&]() { return !queue_.empty() || finished_; });
    }
    if (!queue_.empty()) {
      std::vector<TensorHandle> res = std::move(queue_.front());
      queue_.pop();
      return res;
    }
    CHECK(finished_) << "internal error!";
    return {};
  }

  size_t size() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void set_finished() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_ = true;
    cv_.notify_all();
  }

 private:
  bool finished_{false};
  std::queue<std::vector<TensorHandle>> queue_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

std::vector<std::string> get_data(std::fstream* inf,
                                  const InputMeta& meta,
                                  bool extra = false) {
  CHECK(inf);
  const char* delim = ":";
  std::string tmp_line;
  std::getline(*inf, tmp_line);
  if (inf->eof()) {
    throw EOFException();
  } else if (inf->bad()) {
    LOG(INFO) << "file has bad data.";
  }
  auto tags = split(tmp_line, delim);
  CHECK_EQ(tags[0], meta.alias + "_name");
  if (extra) {
    std::getline(*inf, tmp_line);
    auto type = split(tmp_line, delim);
    CHECK_EQ(type[0], meta.alias + "_type");
    std::getline(*inf, tmp_line);
    auto shape = split(tmp_line, delim);
    CHECK_EQ(shape[0], meta.alias + "_shape");
  }
  std::getline(*inf, tmp_line);
  auto data = split(tmp_line, delim);
  CHECK_EQ(data[0], meta.alias + "_data");
  return split(data[1], ",");
}

std::vector<TensorHandle> next_inputs(const RuntimeContext& ctx,
                                      std::fstream* inf) {
  CHECK(*inf);
  CHECK(!inf->eof()) << "The file is in wrong state.";
  std::vector<TensorHandle> res;
  const auto& cfg = ctx.config();
  const char* split = ",";
  for (auto& id : cfg.ids) {
    res.emplace_back(id, get_data(inf, id), split);
  }
  for (auto& c : cfg.others) {
    res.emplace_back(c, get_data(inf, c, true), split);
  }
  return res;
}

void file_loader(const RuntimeContext& ctx, FeedData* data) {
  CHECK(data) << "The pointer of data must not be null.";
  std::fstream inf;
  inf.open(ctx.config().file_path, std::ios::in);
  CHECK(inf.is_open()) << "file not opening correctly: "
                       << ctx.config().file_path;
  while (true) {
    while (data->size() < ctx.config().capability) {
      try {
        data->push(next_inputs(ctx, &inf));
      } catch (EOFException& e) {
        data->set_finished();
        return;
      }
    }
    std::this_thread::yield();
  }
}

void print_tensor(const paddle::PaddleTensor& tensor) {
  LOG(INFO) << "tensor name: " << tensor.name;
  if (tensor.dtype == paddle_infer::DataType::FLOAT32) {
    auto& buf = tensor.data;
    const auto* ptr = reinterpret_cast<const float*>(buf.data());
    for (int i = 0; i < buf.length() / sizeof(float); ++i) {
      LOG(INFO) << ptr[i];
    }
  } else {
    LOG(FATAL) << "unsupport dtype!";
  }
}
