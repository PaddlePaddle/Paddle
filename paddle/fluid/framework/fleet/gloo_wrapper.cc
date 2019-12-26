#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"

namespace gloo {
namespace rendezvous {

explicit HdfsStore(const std::string& path, const std::string& hdfs_name,
                   const std::string& hdfs_ugi) {
  path_ = path;
  hdfs_name_ = hdfs_name;
  hdfs_ugi_ = hdfs_ugi;
  std::string cmd = std::string("hadoop fs");
  cmd += " -D fs.default.name=" + fs_name;
  cmd += " -D hadoop.job.ugi=" + fs_ugi;
  paddle::framework::hdfs_set_command(cmd);
}

void HdfsStore::Set(const std::string& key, const std::vector<char>& data) {
  auto tmp = TmpPath(key);
  auto path = ObjectPath(key);
  bool is_exists = paddle::framework::hdfs_exists(path);
  PADDLE_ENFORCE_EQ(is_exists, false, "path exists: " + path);
  int err_no = 0;
  std::shared_ptr<FILE> fp = paddle::framework::fs_open_write(tmp, &err_no, "");
  size_t write_count =
      fwrite_unlocked(data.data(), 1, data.size(), fp.get());
  paddle::framework::fs_mv(tmp, path)
}

std::vector<char> HdfsStore::Get(const std::string& key) {
  auto path = ObjectPath(key);
  std::vector<char> result;
  // block until key is set
  wait({key});
  bool is_exists = paddle::framework::hdfs_exists(path);
  PADDLE_ENFORCE_EQ(is_exists, true, "path not exists: " + path);
  int err_no = 0;
  std::shared_ptr<FILE> fp = fs_open_read(path, &err_no, "");
  fseek(fp, 0 , SEEK_END);
  size_t size = ftell (pFile);
  rewind(fp);
  result.resize(size);
  fread(result.data(), 1, size, fp.get());
  return result;
}

void HdfsStore::Wait(const std::vector<std::string>& keys) {
  const auto start = std::chrono::steady_clock::now();
}

void HdfsStore::Wait(const std::vector<std::string>& keys,
          const std::chrono::milliseconds& timeout) {
  auto start = std::chrono::steady_clock::now();
  while (!Check(keys)) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      PADDLE_ENFORCE_EQ(0, 1, "Wait timeout for key(s): " + ::gloo::MakeString(keys));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

std::string HdfsStore::EncodeName(const std::string& name) {
  thread_local std::hash<std::string> hash_func;
  return std::to_string(hash_func(name));
}

std::string HdfsStore::TmpPath(const std::string& name) {
  return path_ + "/." + EncodeName(name) + "_tmp";
}

std::string HdfsStore::ObjectPath(const std::string& name) {
  return path_ + "/." + EncodeName(name);
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

}
}


namespace paddle {
namespace framework {

template void GlooWrapper::AllReduce<int64_t>(
    const std::vector<int64_t>& sendbuf, std::vector<int64_t>& recvbuf);
template void GlooWrapper::AllReduce<double>(
    const std::vector<double>& sendbuf, std::vector<double>& recvbuf);
template std::vector<int64_t> GlooWrapper::AllGather<int64_t>(
    const int64_t& input);
template std::vector<double> GlooWrapper::AllGather<double>(
    const double& input);

}
}
