#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <string>

#include <gloo/allreduce.h>
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/allgather.h>
#include "paddle/fluid/framework/variable_helper.h"

namespace gloo {
namespace rendezvous {

class HdfsStore : public gloo::rendezvous::Store{
public:
  explicit HdfsStore(const std::string& path, const std::string& hdfs_name,
                     const std::string& hdfs_ugi) {
    path_ = path;
    hdfs_name_ = hdfs_name;
    hdfs_ugi_ = hdfs_ugi;
  }

  virtual ~HdfsStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data) override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override;

  virtual void wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout) override;

/*
  virtual void set(const std::string& key, const std::vector<char>& data) {
    VLOG(0) << "HdfsStore::set";
  }

  virtual std::vector<char> get(const std::string& key) {
    VLOG(0) << "HdfsStore::get";
    return std::vector<char>();
  }

  virtual void wait(const std::vector<std::string>& keys) {
    VLOG(0) << "HdfsStore::wait";
  }*/
protected:
  std::string hdfs_name_;
  std::string hdfs_ugi_;
  std::string path_;
};

}
}


namespace paddle {
namespace framework {

class GlooWrapper {
public:

  GlooWrapper() {}
  virtual ~GlooWrapper() {}
  
  void Init(int rank, int size, const std::string& path){ //gloo::rendezvous::Store store) {
    if (is_initialized_) {
        return;
    }
    this->rank = rank;
    this->size = size;
    gloo::transport::tcp::attr attr;
    attr.iface = "lo";//"eth0";
    auto fileStore = gloo::rendezvous::HdfsStore(path);//gloo::rendezvous::FileStore(path);
    auto prefixStore = gloo::rendezvous::PrefixStore("123", fileStore);
    auto dev = gloo::transport::tcp::CreateDevice(attr);//("localhost");//attr);
    auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
    context->connectFullMesh(prefixStore, dev);
    this->kContext = std::move(context);
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
  void AllReduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) { //, int mode) {
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
  std::vector<T> AllGather(const T& input) { //const std::vector<T>& sendbuf, std::vector<T>& recvbuf) {
    CHECK_EQ(is_initialized_, true);
    //CHECK_EQ(sendbuf.size() == recvbuf.size(), true);
    std::vector<T> ret(size, T());
    gloo::AllgatherOptions opts(kContext);
    opts.setInput(const_cast<T*>(&input), 1);
    opts.setOutput(ret.data(), size);
    gloo::allgather(opts);
    return std::move(ret);
  }

/*
  int AllReduce(const vector<int>& sendbuf, vector<int>& recvbuf, int mode) {
    CHECK_EQ(sendbuf.size() == recvbuf.size());
    gloo::AllreduceOptions opts(kContext);
    opts.setInput(const_cast<int*>((const int*) sendbuf.data()), sendbuf.size());
  }

  int AllReduce(const vector<float>& sendbuf, vector<float>& recvbuf, int mode) {
    CHECK_EQ(sendbuf.size() == recvbuf.size());
    gloo::AllreduceOptions opts(kContext);
    opts.setInput(const_cast<float*>((const float*) sendbuf.data()), sendbuf.size());
  }

  int AllReduce(const vector<double>& sendbuf, vector<double>& recvbuf, int mode) {
    CHECK_EQ(sendbuf.size() == recvbuf.size());
    gloo::AllreduceOptions opts(kContext);
    opts.setInput(const_cast<double*>((const double*) sendbuf.data()), sendbuf.size());
  }
  */

/*  static std::shared_ptr<GlooWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::GlooWrapper());
    }
    return s_instance_;
  }
  */

protected:
//  static std::shared_ptr<GlooWrapper> s_instance_;
  bool is_initialized_ =false;
  std::shared_ptr<gloo::Context> kContext;

  int rank;
  int size;

};

}
}
