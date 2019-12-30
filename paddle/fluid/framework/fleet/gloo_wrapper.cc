/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"

namespace gloo {
namespace rendezvous {

HdfsStore::HdfsStore(const std::string& path) {
  path_ = path;
  wait_sleep_ms = 100;
}

void HdfsStore::set(const std::string& key, const std::vector<char>& data) {
  auto tmp = TmpPath(key);
  auto path = ObjectPath(key);
  bool is_exists = paddle::framework::hdfs_exists(path);
  PADDLE_ENFORCE_EQ(is_exists, false, "path exists: " + path);
  int err_no = 0;
  std::shared_ptr<FILE> fp = paddle::framework::fs_open_write(tmp, &err_no, "");
  size_t write_count = fwrite_unlocked(data.data(), 1, data.size(), fp.get());
  VLOG(3) << "HdfsStore::set write_count=" << write_count;
  fp.reset();
  paddle::framework::fs_mv(tmp, path);
}

std::vector<char> HdfsStore::get(const std::string& key) {
  auto path = ObjectPath(key);
  std::vector<char> result;
  // block until key is set
  wait({key});
  bool is_exists = paddle::framework::hdfs_exists(path);
  PADDLE_ENFORCE_EQ(is_exists, true, "path not exists: " + path);
  int err_no = 0;
  std::shared_ptr<FILE> fp = paddle::framework::fs_open_read(path, &err_no, "");
  char buffer = '\0';
  size_t read_count = 0;
  while (fread(&buffer, 1, 1, fp.get()) == 1) {
    ++read_count;
    result.push_back(buffer);
  }
  VLOG(3) << "HdfsStore::get read_count " << read_count;
  return result;
}

void HdfsStore::wait(const std::vector<std::string>& keys) {
  wait(keys, gloo::rendezvous::Store::kDefaultTimeout);
}

void HdfsStore::wait(const std::vector<std::string>& keys,
          const std::chrono::milliseconds& timeout) {
  auto start = std::chrono::steady_clock::now();
  while (!Check(keys)) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      PADDLE_ENFORCE_EQ(0, 1,
                        "Wait timeout for key(s): " + ::gloo::MakeString(keys));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_sleep_ms));
  }
}

std::string HdfsStore::EncodeName(const std::string& name) {
  thread_local std::hash<std::string> hash_func;
  return std::to_string(hash_func(name));
}

std::string HdfsStore::TmpPath(const std::string& name) {
  return path_ + "/" + EncodeName(name) + "_tmp";
}

std::string HdfsStore::ObjectPath(const std::string& name) {
  return path_ + "/" + EncodeName(name);
}

bool HdfsStore::Check(const std::vector<std::string>& keys) {
  std::vector<std::string> paths;
  for (const auto& key : keys) {
    paths.push_back(ObjectPath(key));
  }
  for (const auto& path : paths) {
    bool is_exists = paddle::framework::hdfs_exists(path);
    if (!is_exists) {
      return false;
    }
  }
  return true;
}

}  // namespace rendezvous
}  // namespace gloo


namespace paddle {
namespace framework {

void GlooWrapper::Init(int rank, int size, const std::string& path,
                       const std::string& fs_name, const std::string& fs_ugi,
                       const std::string& iface, const std::string& prefix) {
  if (is_initialized_) {
    return;
  }
  this->rank = rank;
  this->size = size;
  std::string cmd = std::string("hadoop fs");
  cmd += " -D fs.default.name=" + fs_name;
  cmd += " -D hadoop.job.ugi=" + fs_ugi;
  paddle::framework::hdfs_set_command(cmd);
  gloo::transport::tcp::attr attr;
  attr.iface = iface;
  auto fileStore = gloo::rendezvous::HdfsStore(path);
  auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);
  auto dev = gloo::transport::tcp::CreateDevice(attr);
  auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
  context->connectFullMesh(prefixStore, dev);
  this->kContext = std::move(context);
  is_initialized_ = true;
}

template void GlooWrapper::AllReduce<int64_t>(                            // NOLINT
    const std::vector<int64_t>& sendbuf, std::vector<int64_t>& recvbuf);  // NOLINT
template void GlooWrapper::AllReduce<double>(                             // NOLINT
    const std::vector<double>& sendbuf, std::vector<double>& recvbuf);    // NOLINT
template std::vector<int64_t> GlooWrapper::AllGather<int64_t>(
    const int64_t& input);
template std::vector<double> GlooWrapper::AllGather<double>(
    const double& input);

}  // namespace framework
}  // namespace paddle
