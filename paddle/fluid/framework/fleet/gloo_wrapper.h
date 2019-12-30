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

#pragma once

#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gloo/allreduce.h>
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/allgather.h>
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/framework/io/fs.h"

namespace gloo {
namespace rendezvous {

class HdfsStore : public gloo::rendezvous::Store{
public:
  explicit HdfsStore(const std::string& path);

  virtual ~HdfsStore() {}

  void set(const std::string& key, const std::vector<char>& data) override;

  std::vector<char> get(const std::string& key) override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override;

protected:

  std::string EncodeName(const std::string& name);

  std::string TmpPath(const std::string& name);

  std::string ObjectPath(const std::string& name);

  bool Check(const std::vector<std::string>& keys);

  std::string path_;
  int wait_sleep_ms;
};

}  // namespace rendezvous
}  // namespace gloo


namespace paddle {
namespace framework {

class GlooWrapper {
public:

  GlooWrapper() {}
  virtual ~GlooWrapper() {}
  
  void Init(int rank, int size, const std::string& path, 
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
    auto fileStore = gloo::rendezvous::HdfsStore(path);//, fs_name, fs_ugi);
    auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);
    auto dev = gloo::transport::tcp::CreateDevice(attr);
    auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
    context->connectFullMesh(prefixStore, dev);
    this->kContext = std::move(context);
//    std::string cmd = std::string("hadoop fs");//
//    cmd += " -D fs.default.name=" + fs_name;
//    cmd += " -D hadoop.job.ugi=" + fs_ugi;
//    paddle::framework::hdfs_set_command(cmd);
    is_initialized_ = true;
  }

  int Rank() {
    CHECK_EQ(is_initialized_, true);
    return rank;
  }

  int Size () {
    CHECK_EQ(is_initialized_, true);
    return size;
  }

  void Barrier() {
    CHECK_EQ(is_initialized_, true);
    gloo::BarrierOptions opts(kContext);
    gloo::barrier(opts);
  }

  template<typename T>
  void AllReduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) {  // NOLINT
    CHECK_EQ(is_initialized_, true);
    CHECK_EQ(sendbuf.size() == recvbuf.size(), true);
    gloo::AllreduceOptions opts(kContext);
    opts.setInput(const_cast<T*>((const T*) sendbuf.data()), sendbuf.size());
    opts.setOutput(recvbuf.data(), recvbuf.size());
    opts.setReduceFunction(
        static_cast<void(*)(void*, const void*, const void*, size_t)>(&gloo::sum<T>));
    gloo::allreduce(opts);
  }

  template<typename T>
  std::vector<T> AllGather(const T& input) {
    CHECK_EQ(is_initialized_, true);
    std::vector<T> ret(size, T());
    gloo::AllgatherOptions opts(kContext);
    opts.setInput(const_cast<T*>(&input), 1);
    opts.setOutput(ret.data(), size);
    gloo::allgather(opts);
    return std::move(ret);
  }

protected:
  bool is_initialized_ =false;
  std::shared_ptr<gloo::Context> kContext;

  int rank;
  int size;

};

}  // namespace framework
}  // namespace paddle
