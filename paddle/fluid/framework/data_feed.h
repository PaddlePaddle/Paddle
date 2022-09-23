/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include <fstream>
#include <future>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/string/string_helper.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif

DECLARE_int32(record_pool_max_size);
DECLARE_int32(slotpool_thread_num);
DECLARE_bool(enable_slotpool_wait_release);
DECLARE_bool(enable_slotrecord_reset_shrink);

namespace paddle {
namespace framework {
class DataFeedDesc;
class Scope;
class Variable;
class NeighborSampleResult;
class NodeQueryResult;
}  // namespace framework
}  // namespace paddle

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

// DataFeed is the base virtual class for all ohther DataFeeds.
// It is used to read files and parse the data for subsequent trainer.
// Example:
//   DataFeed* reader =
//   paddle::framework::DataFeedFactory::CreateDataFeed(data_feed_name);
//   reader->Init(data_feed_desc); // data_feed_desc is a protobuf object
//   reader->SetFileList(filelist);
//   const std::vector<std::string> & use_slot_alias =
//   reader->GetUseSlotAlias();
//   for (auto name: use_slot_alias){ // for binding memory
//     reader->AddFeedVar(scope->Var(name), name);
//   }
//   reader->Start();
//   while (reader->Next()) {
//      // trainer do something
//   }

template <typename T>
struct SlotValues {
  std::vector<T> slot_values;
  std::vector<uint32_t> slot_offsets;

  void add_values(const T* values, uint32_t num) {
    if (slot_offsets.empty()) {
      slot_offsets.push_back(0);
    }
    if (num > 0) {
      slot_values.insert(slot_values.end(), values, values + num);
    }
    slot_offsets.push_back(static_cast<uint32_t>(slot_values.size()));
  }
  T* get_values(int idx, size_t* size) {
    uint32_t& offset = slot_offsets[idx];
    (*size) = slot_offsets[idx + 1] - offset;
    return &slot_values[offset];
  }
  void add_slot_feasigns(const std::vector<std::vector<T>>& slot_feasigns,
                         uint32_t fea_num) {
    slot_values.reserve(fea_num);
    int slot_num = static_cast<int>(slot_feasigns.size());
    slot_offsets.resize(slot_num + 1);
    for (int i = 0; i < slot_num; ++i) {
      auto& slot_val = slot_feasigns[i];
      slot_offsets[i] = static_cast<uint32_t>(slot_values.size());
      uint32_t num = static_cast<uint32_t>(slot_val.size());
      if (num > 0) {
        slot_values.insert(slot_values.end(), slot_val.begin(), slot_val.end());
      }
    }
    slot_offsets[slot_num] = slot_values.size();
  }
  void clear(bool shrink) {
    slot_offsets.clear();
    slot_values.clear();
    if (shrink) {
      slot_values.shrink_to_fit();
      slot_offsets.shrink_to_fit();
    }
  }
};
union FeatureFeasign {
  uint64_t uint64_feasign_;
  float float_feasign_;
};

struct FeatureItem {
  FeatureItem() {}
  FeatureItem(FeatureFeasign sign, uint16_t slot) {
    this->sign() = sign;
    this->slot() = slot;
  }
  FeatureFeasign& sign() {
    return *(reinterpret_cast<FeatureFeasign*>(sign_buffer()));
  }
  const FeatureFeasign& sign() const {
    const FeatureFeasign* ret =
        reinterpret_cast<FeatureFeasign*>(sign_buffer());
    return *ret;
  }
  uint16_t& slot() { return slot_; }
  const uint16_t& slot() const { return slot_; }

 private:
  char* sign_buffer() const { return const_cast<char*>(sign_); }
  char sign_[sizeof(FeatureFeasign)];
  uint16_t slot_;
};

struct AllSlotInfo {
  std::string slot;
  std::string type;
  int used_idx;
  int slot_value_idx;
};
struct UsedSlotInfo {
  int idx;
  int slot_value_idx;
  std::string slot;
  std::string type;
  bool dense;
  std::vector<int> local_shape;
  int total_dims_without_inductive;
  int inductive_shape_index;
};
struct SlotRecordObject {
  uint64_t search_id;
  uint32_t rank;
  uint32_t cmatch;
  std::string ins_id_;
  SlotValues<uint64_t> slot_uint64_feasigns_;
  SlotValues<float> slot_float_feasigns_;

  ~SlotRecordObject() { clear(true); }
  void reset(void) { clear(FLAGS_enable_slotrecord_reset_shrink); }
  void clear(bool shrink) {
    slot_uint64_feasigns_.clear(shrink);
    slot_float_feasigns_.clear(shrink);
  }
};
using SlotRecord = SlotRecordObject*;
// sizeof Record is much less than std::vector<MultiSlotType>
struct Record {
  std::vector<FeatureItem> uint64_feasigns_;
  std::vector<FeatureItem> float_feasigns_;
  std::string ins_id_;
  std::string content_;
  uint64_t search_id;
  uint32_t rank;
  uint32_t cmatch;
  std::string uid_;
};

inline SlotRecord make_slotrecord() {
  static const size_t slot_record_byte_size = sizeof(SlotRecordObject);
  void* p = malloc(slot_record_byte_size);
  new (p) SlotRecordObject;
  return reinterpret_cast<SlotRecordObject*>(p);
}

inline void free_slotrecord(SlotRecordObject* p) {
  p->~SlotRecordObject();
  free(p);
}

template <class T>
class SlotObjAllocator {
 public:
  explicit SlotObjAllocator(std::function<void(T*)> deleter)
      : free_nodes_(NULL), capacity_(0), deleter_(deleter) {}
  ~SlotObjAllocator() { clear(); }

  void clear() {
    T* tmp = NULL;
    while (free_nodes_ != NULL) {
      tmp = reinterpret_cast<T*>(reinterpret_cast<void*>(free_nodes_));
      free_nodes_ = free_nodes_->next;
      deleter_(tmp);
      --capacity_;
    }
    CHECK_EQ(capacity_, static_cast<size_t>(0));
  }
  T* acquire(void) {
    T* x = NULL;
    x = reinterpret_cast<T*>(reinterpret_cast<void*>(free_nodes_));
    free_nodes_ = free_nodes_->next;
    --capacity_;
    return x;
  }
  void release(T* x) {
    Node* node = reinterpret_cast<Node*>(reinterpret_cast<void*>(x));
    node->next = free_nodes_;
    free_nodes_ = node;
    ++capacity_;
  }
  size_t capacity(void) { return capacity_; }

 private:
  struct alignas(T) Node {
    union {
      Node* next;
      char data[sizeof(T)];
    };
  };
  Node* free_nodes_;  // a list
  size_t capacity_;
  std::function<void(T*)> deleter_ = nullptr;
};
static const int OBJPOOL_BLOCK_SIZE = 10000;
class SlotObjPool {
 public:
  SlotObjPool()
      : max_capacity_(FLAGS_record_pool_max_size), alloc_(free_slotrecord) {
    ins_chan_ = MakeChannel<SlotRecord>();
    ins_chan_->SetBlockSize(OBJPOOL_BLOCK_SIZE);
    for (int i = 0; i < FLAGS_slotpool_thread_num; ++i) {
      threads_.push_back(std::thread([this]() { run(); }));
    }
    disable_pool_ = false;
    count_ = 0;
  }
  ~SlotObjPool() {
    ins_chan_->Close();
    for (auto& t : threads_) {
      t.join();
    }
  }
  void disable_pool(bool disable) { disable_pool_ = disable; }
  void set_max_capacity(size_t max_capacity) { max_capacity_ = max_capacity; }
  void get(std::vector<SlotRecord>* output, int n) {
    output->resize(n);
    return get(&(*output)[0], n);
  }
  void get(SlotRecord* output, int n) {
    int size = 0;
    mutex_.lock();
    int left = static_cast<int>(alloc_.capacity());
    if (left > 0) {
      size = (left >= n) ? n : left;
      for (int i = 0; i < size; ++i) {
        output[i] = alloc_.acquire();
      }
    }
    mutex_.unlock();
    count_ += n;
    if (size == n) {
      return;
    }
    for (int i = size; i < n; ++i) {
      output[i] = make_slotrecord();
    }
  }
  void put(std::vector<SlotRecord>* input) {
    size_t size = input->size();
    if (size == 0) {
      return;
    }
    put(&(*input)[0], size);
    input->clear();
  }
  void put(SlotRecord* input, size_t size) {
    CHECK(ins_chan_->WriteMove(size, input) == size);
  }
  void run(void) {
    std::vector<SlotRecord> input;
    while (ins_chan_->ReadOnce(input, OBJPOOL_BLOCK_SIZE)) {
      if (input.empty()) {
        continue;
      }
      // over max capacity
      size_t n = input.size();
      count_ -= n;
      if (disable_pool_ || n + capacity() > max_capacity_) {
        for (auto& t : input) {
          free_slotrecord(t);
        }
      } else {
        for (auto& t : input) {
          t->reset();
        }
        mutex_.lock();
        for (auto& t : input) {
          alloc_.release(t);
        }
        mutex_.unlock();
      }
      input.clear();
    }
  }
  void clear(void) {
    platform::Timer timeline;
    timeline.Start();
    mutex_.lock();
    alloc_.clear();
    mutex_.unlock();
    // wait release channel data
    if (FLAGS_enable_slotpool_wait_release) {
      while (!ins_chan_->Empty()) {
        sleep(1);
      }
    }
    timeline.Pause();
    VLOG(3) << "clear slot pool data size=" << count_.load()
            << ", span=" << timeline.ElapsedSec();
  }
  size_t capacity(void) {
    mutex_.lock();
    size_t total = alloc_.capacity();
    mutex_.unlock();
    return total;
  }

 private:
  size_t max_capacity_;
  Channel<SlotRecord> ins_chan_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  SlotObjAllocator<SlotRecordObject> alloc_;
  bool disable_pool_;
  std::atomic<long> count_;  // NOLINT
};

inline SlotObjPool& SlotRecordPool() {
  static SlotObjPool pool;
  return pool;
}
struct PvInstanceObject {
  std::vector<Record*> ads;
  void merge_instance(Record* ins) { ads.push_back(ins); }
};

using PvInstance = PvInstanceObject*;

inline PvInstance make_pv_instance() { return new PvInstanceObject(); }

struct SlotConf {
  std::string name;
  std::string type;
  int use_slots_index;
  int use_slots_is_dense;
};

class CustomParser {
 public:
  CustomParser() {}
  virtual ~CustomParser() {}
  virtual void Init(const std::vector<SlotConf>& slots) = 0;
  virtual bool Init(const std::vector<AllSlotInfo>& slots) = 0;
  virtual void ParseOneInstance(const char* str, Record* instance) = 0;
  virtual int ParseInstance(int len,
                            const char* str,
                            std::vector<Record>* instances) {
    return 0;
  };
  virtual bool ParseOneInstance(
      const std::string& line,
      std::function<void(std::vector<SlotRecord>&, int)>
          GetInsFunc) {  // NOLINT
    return true;
  }
  virtual bool ParseFileInstance(
      std::function<int(char* buf, int len)> ReadBuffFunc,
      std::function<void(std::vector<SlotRecord>&, int, int)>
          PullRecordsFunc,  // NOLINT
      int& lines) {         // NOLINT
    return false;
  }
};

struct UsedSlotGpuType {
  int is_uint64_value;
  int slot_value_idx;
};

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
template <typename T>
struct CudaBuffer {
  T* cu_buffer;
  uint64_t buf_size;

  CudaBuffer<T>() {
    cu_buffer = NULL;
    buf_size = 0;
  }
  ~CudaBuffer<T>() { free(); }
  T* data() { return cu_buffer; }
  uint64_t size() { return buf_size; }
  void malloc(uint64_t size) {
    buf_size = size;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&cu_buffer), size * sizeof(T)));
  }
  void free() {
    if (cu_buffer != NULL) {
      CUDA_CHECK(cudaFree(cu_buffer));
      cu_buffer = NULL;
    }
    buf_size = 0;
  }
  void resize(uint64_t size) {
    if (size <= buf_size) {
      return;
    }
    free();
    malloc(size);
  }
};
template <typename T>
struct HostBuffer {
  T* host_buffer;
  size_t buf_size;
  size_t data_len;

  HostBuffer<T>() {
    host_buffer = NULL;
    buf_size = 0;
    data_len = 0;
  }
  ~HostBuffer<T>() { free(); }

  T* data() { return host_buffer; }
  const T* data() const { return host_buffer; }
  size_t size() const { return data_len; }
  void clear() { free(); }
  T& back() { return host_buffer[data_len - 1]; }

  T& operator[](size_t i) { return host_buffer[i]; }
  const T& operator[](size_t i) const { return host_buffer[i]; }
  void malloc(size_t len) {
    buf_size = len;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_buffer),
                             buf_size * sizeof(T),
                             cudaHostAllocDefault));
    CHECK(host_buffer != NULL);
  }
  void free() {
    if (host_buffer != NULL) {
      CUDA_CHECK(cudaFreeHost(host_buffer));
      host_buffer = NULL;
    }
    buf_size = 0;
  }
  void resize(size_t size) {
    if (size <= buf_size) {
      data_len = size;
      return;
    }
    data_len = size;
    free();
    malloc(size);
  }
};

struct BatchCPUValue {
  HostBuffer<int> h_uint64_lens;
  HostBuffer<uint64_t> h_uint64_keys;
  HostBuffer<int> h_uint64_offset;

  HostBuffer<int> h_float_lens;
  HostBuffer<float> h_float_keys;
  HostBuffer<int> h_float_offset;

  HostBuffer<int> h_rank;
  HostBuffer<int> h_cmatch;
  HostBuffer<int> h_ad_offset;
};

struct BatchGPUValue {
  CudaBuffer<int> d_uint64_lens;
  CudaBuffer<uint64_t> d_uint64_keys;
  CudaBuffer<int> d_uint64_offset;

  CudaBuffer<int> d_float_lens;
  CudaBuffer<float> d_float_keys;
  CudaBuffer<int> d_float_offset;

  CudaBuffer<int> d_rank;
  CudaBuffer<int> d_cmatch;
  CudaBuffer<int> d_ad_offset;
};

class MiniBatchGpuPack {
 public:
  MiniBatchGpuPack(const paddle::platform::Place& place,
                   const std::vector<UsedSlotInfo>& infos);
  ~MiniBatchGpuPack();
  void reset(const paddle::platform::Place& place);
  void pack_instance(const SlotRecord* ins_vec, int num);
  int ins_num() { return ins_num_; }
  int pv_num() { return pv_num_; }
  BatchGPUValue& value() { return value_; }
  BatchCPUValue& cpu_value() { return buf_; }
  UsedSlotGpuType* get_gpu_slots(void) {
    return reinterpret_cast<UsedSlotGpuType*>(gpu_slots_.data());
  }
  SlotRecord* get_records(void) { return &ins_vec_[0]; }

  // tensor gpu memory reused
  void resize_tensor(void) {
    if (used_float_num_ > 0) {
      int float_total_len = buf_.h_float_lens.back();
      if (float_total_len > 0) {
        float_tensor_.mutable_data<float>({float_total_len, 1}, this->place_);
      }
    }
    if (used_uint64_num_ > 0) {
      int uint64_total_len = buf_.h_uint64_lens.back();
      if (uint64_total_len > 0) {
        uint64_tensor_.mutable_data<int64_t>({uint64_total_len, 1},
                                             this->place_);
      }
    }
  }
  LoDTensor& float_tensor(void) { return float_tensor_; }
  LoDTensor& uint64_tensor(void) { return uint64_tensor_; }

  HostBuffer<size_t>& offsets(void) { return offsets_; }
  HostBuffer<void*>& h_tensor_ptrs(void) { return h_tensor_ptrs_; }

  void* gpu_slot_offsets(void) { return gpu_slot_offsets_->ptr(); }

  void* slot_buf_ptr(void) { return slot_buf_ptr_->ptr(); }

  void resize_gpu_slot_offsets(const size_t slot_total_bytes) {
    if (gpu_slot_offsets_ == nullptr) {
      gpu_slot_offsets_ = memory::AllocShared(place_, slot_total_bytes);
    } else if (gpu_slot_offsets_->size() < slot_total_bytes) {
      auto buf = memory::AllocShared(place_, slot_total_bytes);
      gpu_slot_offsets_.swap(buf);
      buf = nullptr;
    }
  }
  const std::string& get_lineid(int idx) {
    if (enable_pv_) {
      return ins_vec_[idx]->ins_id_;
    }
    return batch_ins_[idx]->ins_id_;
  }

 private:
  void transfer_to_gpu(void);
  void pack_all_data(const SlotRecord* ins_vec, int num);
  void pack_uint64_data(const SlotRecord* ins_vec, int num);
  void pack_float_data(const SlotRecord* ins_vec, int num);

 public:
  template <typename T>
  void copy_host2device(CudaBuffer<T>* buf, const T* val, size_t size) {
    if (size == 0) {
      return;
    }
    buf->resize(size);
    CUDA_CHECK(cudaMemcpyAsync(
        buf->data(), val, size * sizeof(T), cudaMemcpyHostToDevice, stream_));
  }
  template <typename T>
  void copy_host2device(CudaBuffer<T>* buf, const HostBuffer<T>& val) {
    copy_host2device(buf, val.data(), val.size());
  }

 private:
  paddle::platform::Place place_;
  cudaStream_t stream_;
  BatchGPUValue value_;
  BatchCPUValue buf_;
  int ins_num_ = 0;
  int pv_num_ = 0;

  bool enable_pv_ = false;
  int used_float_num_ = 0;
  int used_uint64_num_ = 0;
  int used_slot_size_ = 0;

  CudaBuffer<UsedSlotGpuType> gpu_slots_;
  std::vector<UsedSlotGpuType> gpu_used_slots_;
  std::vector<SlotRecord> ins_vec_;
  const SlotRecord* batch_ins_ = nullptr;

  // uint64 tensor
  LoDTensor uint64_tensor_;
  // float tensor
  LoDTensor float_tensor_;
  // batch
  HostBuffer<size_t> offsets_;
  HostBuffer<void*> h_tensor_ptrs_;

  std::shared_ptr<phi::Allocation> gpu_slot_offsets_ = nullptr;
  std::shared_ptr<phi::Allocation> slot_buf_ptr_ = nullptr;
};
class MiniBatchGpuPackMgr {
  static const int MAX_DEIVCE_NUM = 16;

 public:
  MiniBatchGpuPackMgr() {
    for (int i = 0; i < MAX_DEIVCE_NUM; ++i) {
      pack_list_[i] = nullptr;
    }
  }
  ~MiniBatchGpuPackMgr() {
    for (int i = 0; i < MAX_DEIVCE_NUM; ++i) {
      if (pack_list_[i] == nullptr) {
        continue;
      }
      delete pack_list_[i];
      pack_list_[i] = nullptr;
    }
  }
  // one device one thread
  MiniBatchGpuPack* get(const paddle::platform::Place& place,
                        const std::vector<UsedSlotInfo>& infos) {
    int device_id = place.GetDeviceId();
    if (pack_list_[device_id] == nullptr) {
      pack_list_[device_id] = new MiniBatchGpuPack(place, infos);
    } else {
      pack_list_[device_id]->reset(place);
    }
    return pack_list_[device_id];
  }

 private:
  MiniBatchGpuPack* pack_list_[MAX_DEIVCE_NUM];
};
// global mgr
inline MiniBatchGpuPackMgr& BatchGpuPackMgr() {
  static MiniBatchGpuPackMgr mgr;
  return mgr;
}
#endif

typedef paddle::framework::CustomParser* (*CreateParserObjectFunc)();

class DLManager {
  struct DLHandle {
    void* module;
    paddle::framework::CustomParser* parser;
  };

 public:
  DLManager() {}

  ~DLManager() {
#ifdef _LINUX
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = handle_map_.begin(); it != handle_map_.end(); ++it) {
      delete it->second.parser;
      dlclose(it->second.module);
    }
#endif
  }

  bool Close(const std::string& name) {
#ifdef _LINUX
    auto it = handle_map_.find(name);
    if (it == handle_map_.end()) {
      return true;
    }
    delete it->second.parser;
    dlclose(it->second.module);
#endif
    VLOG(0) << "Not implement in windows";
    return false;
  }

  paddle::framework::CustomParser* Load(const std::string& name,
                                        const std::vector<SlotConf>& conf) {
#ifdef _LINUX
    std::lock_guard<std::mutex> lock(mutex_);
    DLHandle handle;
    std::map<std::string, DLHandle>::iterator it = handle_map_.find(name);
    if (it != handle_map_.end()) {
      return it->second.parser;
    }

    handle.module = dlopen(name.c_str(), RTLD_NOW);
    if (handle.module == nullptr) {
      VLOG(0) << "Create so of " << name << " fail, " << dlerror();
      return nullptr;
    }

    CreateParserObjectFunc create_parser_func =
        (CreateParserObjectFunc)dlsym(handle.module, "CreateParserObject");
    handle.parser = create_parser_func();
    handle.parser->Init(conf);
    handle_map_.insert({name, handle});

    return handle.parser;
#endif
    VLOG(0) << "Not implement in windows";
    return nullptr;
  }

  paddle::framework::CustomParser* Load(const std::string& name,
                                        const std::vector<AllSlotInfo>& conf) {
#ifdef _LINUX
    std::lock_guard<std::mutex> lock(mutex_);
    DLHandle handle;
    std::map<std::string, DLHandle>::iterator it = handle_map_.find(name);
    if (it != handle_map_.end()) {
      return it->second.parser;
    }
    handle.module = dlopen(name.c_str(), RTLD_NOW);
    if (handle.module == nullptr) {
      VLOG(0) << "Create so of " << name << " fail";
      exit(-1);
      return nullptr;
    }

    CreateParserObjectFunc create_parser_func =
        (CreateParserObjectFunc)dlsym(handle.module, "CreateParserObject");
    handle.parser = create_parser_func();
    handle.parser->Init(conf);
    handle_map_.insert({name, handle});

    return handle.parser;
#endif
    VLOG(0) << "Not implement in windows";
    return nullptr;
  }

  paddle::framework::CustomParser* ReLoad(const std::string& name,
                                          const std::vector<SlotConf>& conf) {
    Close(name);
    return Load(name, conf);
  }

 private:
  std::mutex mutex_;
  std::map<std::string, DLHandle> handle_map_;
};

struct engine_wrapper_t {
  std::default_random_engine engine;
#if !defined(_WIN32)
  engine_wrapper_t() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
    static std::atomic<uint64_t> x(0);
    std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
    engine.seed(sseq);
  }
#endif
};

struct BufState {
  int left;
  int right;
  int central_word;
  int step;
  engine_wrapper_t random_engine_;

  int len;
  int cursor;
  int row_num;

  int batch_size;
  int walk_len;
  std::vector<int>* window;

  BufState() {}
  ~BufState() {}

  void Init(int graph_batch_size,
            int graph_walk_len,
            std::vector<int>* graph_window) {
    batch_size = graph_batch_size;
    walk_len = graph_walk_len;
    window = graph_window;

    left = 0;
    right = window->size() - 1;
    central_word = -1;
    step = -1;

    len = 0;
    cursor = 0;
    row_num = 0;
    for (size_t i = 0; i < graph_window->size(); i++) {
      VLOG(2) << "graph_window[" << i << "] = " << (*graph_window)[i];
    }
  }

  void Reset(int total_rows) {
    cursor = 0;
    row_num = total_rows;
    int tmp_len = cursor + batch_size > row_num ? row_num - cursor : batch_size;
    len = tmp_len;
    central_word = -1;
    step = -1;
    GetNextCentrolWord();
  }

  int GetNextStep() {
    step++;
    if (step <= right && central_word + (*window)[step] < walk_len) {
      return 1;
    }
    return 0;
  }

  void Debug() {
    VLOG(2) << "left: " << left << " right: " << right
            << " central_word: " << central_word << " step: " << step
            << " cursor: " << cursor << " len: " << len
            << " row_num: " << row_num;
  }

  int GetNextCentrolWord() {
    if (++central_word >= walk_len) {
      return 0;
    }
    int window_size = window->size() / 2;
    int random_window = random_engine_.engine() % window_size + 1;
    left = window_size - random_window;
    right = window_size + random_window - 1;
    VLOG(2) << "random window: " << random_window << " window[" << left
            << "] = " << (*window)[left] << " window[" << right
            << "] = " << (*window)[right];

    for (step = left; step <= right; step++) {
      if (central_word + (*window)[step] >= 0) {
        return 1;
      }
    }
    return 0;
  }

  int GetNextBatch() {
    cursor += len;
    int tmp_len = cursor + batch_size > row_num ? row_num - cursor : batch_size;
    if (tmp_len == 0) {
      return 0;
    }
    len = tmp_len;
    central_word = -1;
    step = -1;
    GetNextCentrolWord();
    return tmp_len != 0;
  }
};

class GraphDataGenerator {
 public:
  GraphDataGenerator(){};
  virtual ~GraphDataGenerator(){};
  void SetConfig(const paddle::framework::DataFeedDesc& data_feed_desc);
  void AllocResource(const paddle::platform::Place& place,
                     std::vector<LoDTensor*> feed_vec);
  int AcquireInstance(BufState* state);
  int GenerateBatch();
  int FillWalkBuf(std::shared_ptr<phi::Allocation> d_walk);
  int FillFeatureBuf(uint64_t* d_walk, uint64_t* d_feature, size_t key_num);
  int FillFeatureBuf(std::shared_ptr<phi::Allocation> d_walk,
                     std::shared_ptr<phi::Allocation> d_feature);
  void FillOneStep(uint64_t* start_ids,
                   uint64_t* walk,
                   int len,
                   NeighborSampleResult& sample_res,
                   int cur_degree,
                   int step,
                   int* len_per_row);
  int FillInsBuf();
  void SetDeviceKeys(std::vector<uint64_t>* device_keys, int type) {
    type_to_index_[type] = h_device_keys_.size();
    h_device_keys_.push_back(device_keys);
  }
  std::vector<std::shared_ptr<phi::Allocation>> SampleNeighbors(
      int64_t* uniq_nodes, int len, int sample_size,
      std::vector<int>& edges_split_num, int64_t* neighbor_len);

  std::shared_ptr<phi::Allocation> GetReindexResult(
      int64_t* reindex_src_data, const int64_t* center_nodes,
      int* final_nodes_len, int node_len, int64_t neighbor_len);

  std::shared_ptr<phi::Allocation> GenerateSampleGraph(
      uint64_t* node_ids, int len, int* uniq_len, phi::DenseTensor* inverse);

 protected:
  int walk_degree_;
  int walk_len_;
  int window_;
  int once_sample_startid_len_;
  int gpuid_;
  // start ids
  // int64_t* device_keys_;
  // size_t device_key_size_;
  std::vector<std::vector<uint64_t>*> h_device_keys_;
  std::unordered_map<int, int> type_to_index_;
  // point to device_keys_
  size_t cursor_;
  size_t jump_rows_;
  int edge_to_id_len_;
  int uniq_instance_;
  int64_t* id_tensor_ptr_;
  int* index_tensor_ptr_;
  int64_t* show_tensor_ptr_;
  int64_t* clk_tensor_ptr_;

  cudaStream_t stream_;
  paddle::platform::Place place_;
  std::vector<LoDTensor*> feed_vec_;
  std::vector<size_t> offset_;
  std::shared_ptr<phi::Allocation> d_prefix_sum_;
  std::vector<std::shared_ptr<phi::Allocation>> d_device_keys_;

  std::shared_ptr<phi::Allocation> d_walk_;
  std::shared_ptr<phi::Allocation> d_feature_;
  std::shared_ptr<phi::Allocation> d_len_per_row_;
  std::shared_ptr<phi::Allocation> d_random_row_;
  std::shared_ptr<phi::Allocation> d_slot_feature_num_map_;
  std::shared_ptr<phi::Allocation> d_actual_slot_id_map_;
  std::shared_ptr<phi::Allocation> d_fea_offset_map_;
  //
  std::vector<std::shared_ptr<phi::Allocation>> d_sampleidx2rows_;
  int cur_sampleidx2row_;
  // record the keys to call graph_neighbor_sample
  std::shared_ptr<phi::Allocation> d_sample_keys_;
  int sample_keys_len_;

  std::set<int> finish_node_type_;
  std::unordered_map<int, size_t> node_type_start_;
  std::vector<int> infer_node_type_start_;

  std::shared_ptr<phi::Allocation> d_ins_buf_;
  std::shared_ptr<phi::Allocation> d_feature_buf_;
  std::shared_ptr<phi::Allocation> d_pair_num_;
  std::shared_ptr<phi::Allocation> d_slot_tensor_ptr_;
  std::shared_ptr<phi::Allocation> d_slot_lod_tensor_ptr_;
  std::shared_ptr<phi::Allocation> d_reindex_table_key_;
  std::shared_ptr<phi::Allocation> d_reindex_table_value_;
  std::shared_ptr<phi::Allocation> d_reindex_table_index_;
  std::vector<std::shared_ptr<phi::Allocation>> edge_type_graph_;
  int64_t reindex_table_size_;
  int ins_buf_pair_len_;
  // size of a d_walk buf
  size_t buf_size_;
  int repeat_time_;
  std::vector<int> window_step_;
  BufState buf_state_;
  int batch_size_;
  int slot_num_;
  std::vector<int> h_slot_feature_num_map_;
  int fea_num_per_node_;
  int shuffle_seed_;
  int debug_mode_;
  std::vector<int> first_node_type_;
  std::vector<std::vector<int>> meta_path_;
  bool gpu_graph_training_;
  bool sage_mode_;
  std::vector<int> samples_;
};

class DataFeed {
 public:
  DataFeed() {
    mutex_for_pick_file_ = nullptr;
    file_idx_ = nullptr;
    mutex_for_fea_num_ = nullptr;
    total_fea_num_ = nullptr;
  }
  virtual ~DataFeed() {}
  virtual void Init(const DataFeedDesc& data_feed_desc) = 0;
  virtual bool CheckFile(const char* filename) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This function(CheckFile) is not implemented."));
  }
  // Set filelist for DataFeed.
  // Pay attention that it must init all readers before call this function.
  // Otherwise, Init() function will init finish_set_filelist_ flag.
  virtual bool SetFileList(const std::vector<std::string>& files);
  virtual bool Start() = 0;

  // The trainer calls the Next() function, and the DataFeed will load a new
  // batch to the feed_vec. The return value of this function is the batch
  // size of the current batch.
  virtual int Next() = 0;
  // Get all slots' alias which defined in protofile
  virtual const std::vector<std::string>& GetAllSlotAlias() {
    return all_slots_;
  }
  // Get used slots' alias which defined in protofile
  virtual const std::vector<std::string>& GetUseSlotAlias() {
    return use_slots_;
  }
  // This function is used for binding feed_vec memory
  virtual void AddFeedVar(Variable* var, const std::string& name);

  // This function is used for binding feed_vec memory in a given scope
  virtual void AssignFeedVar(const Scope& scope);

  // This function will do nothing at default
  virtual void SetInputPvChannel(void* channel) {}
  // This function will do nothing at default
  virtual void SetOutputPvChannel(void* channel) {}
  // This function will do nothing at default
  virtual void SetConsumePvChannel(void* channel) {}

  // This function will do nothing at default
  virtual void SetInputChannel(void* channel) {}
  // This function will do nothing at default
  virtual void SetOutputChannel(void* channel) {}
  // This function will do nothing at default
  virtual void SetConsumeChannel(void* channel) {}
  // This function will do nothing at default
  virtual void SetThreadId(int thread_id) {}
  // This function will do nothing at default
  virtual void SetThreadNum(int thread_num) {}
  // This function will do nothing at default
  virtual void SetParseInsId(bool parse_ins_id) {}
  virtual void SetParseUid(bool parse_uid) {}
  virtual void SetParseContent(bool parse_content) {}
  virtual void SetParseLogKey(bool parse_logkey) {}
  virtual void SetEnablePvMerge(bool enable_pv_merge) {}
  virtual void SetCurrentPhase(int current_phase) {}
  virtual void SetDeviceKeys(std::vector<uint64_t>* device_keys, int type) {
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
    gpu_graph_data_generator_.SetDeviceKeys(device_keys, type);
#endif
  }
  virtual void SetGpuGraphMode(int gpu_graph_mode) {
    gpu_graph_mode_ = gpu_graph_mode;
  }
  virtual void SetFileListMutex(std::mutex* mutex) {
    mutex_for_pick_file_ = mutex;
  }
  virtual void SetFeaNumMutex(std::mutex* mutex) { mutex_for_fea_num_ = mutex; }
  virtual void SetFileListIndex(size_t* file_index) { file_idx_ = file_index; }
  virtual void SetFeaNum(uint64_t* fea_num) { total_fea_num_ = fea_num; }
  virtual const std::vector<std::string>& GetInsIdVec() const {
    return ins_id_vec_;
  }
  virtual const std::vector<std::string>& GetInsContentVec() const {
    return ins_content_vec_;
  }
  virtual int GetCurBatchSize() { return batch_size_; }
  virtual bool IsTrainMode() { return train_mode_; }
  virtual void LoadIntoMemory() {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This function(LoadIntoMemory) is not implemented."));
  }
  virtual void SetPlace(const paddle::platform::Place& place) {
    place_ = place;
  }
  virtual const paddle::platform::Place& GetPlace() const { return place_; }

 protected:
  // The following three functions are used to check if it is executed in this
  // order:
  //   Init() -> SetFileList() -> Start() -> Next()
  virtual void CheckInit();
  virtual void CheckSetFileList();
  virtual void CheckStart();
  virtual void SetBatchSize(
      int batch);  // batch size will be set in Init() function
  // This function is used to pick one file from the global filelist(thread
  // safe).
  virtual bool PickOneFile(std::string* filename);
  virtual void CopyToFeedTensor(void* dst, const void* src, size_t size);

  std::vector<std::string> filelist_;
  size_t* file_idx_;
  std::mutex* mutex_for_pick_file_;
  std::mutex* mutex_for_fea_num_ = nullptr;
  uint64_t* total_fea_num_ = nullptr;
  uint64_t fea_num_ = 0;

  // the alias of used slots, and its order is determined by
  // data_feed_desc(proto object)
  std::vector<std::string> use_slots_;
  std::vector<bool> use_slots_is_dense_;

  // the alias of all slots, and its order is determined by data_feed_desc(proto
  // object)
  std::vector<std::string> all_slots_;
  std::vector<std::string> all_slots_type_;
  std::vector<std::vector<int>> use_slots_shape_;
  std::vector<int> inductive_shape_index_;
  std::vector<int> total_dims_without_inductive_;
  // For the inductive shape passed within data
  std::vector<std::vector<int>> multi_inductive_shape_index_;
  std::vector<int>
      use_slots_index_;  // -1: not used; >=0: the index of use_slots_

  // The data read by DataFeed will be stored here
  std::vector<LoDTensor*> feed_vec_;

  LoDTensor* rank_offset_;

  // the batch size defined by user
  int default_batch_size_;
  // current batch size
  int batch_size_;

  bool finish_init_;
  bool finish_set_filelist_;
  bool finish_start_;
  std::string pipe_command_;
  std::string so_parser_name_;
  std::vector<SlotConf> slot_conf_;
  std::vector<std::string> ins_id_vec_;
  std::vector<std::string> ins_content_vec_;
  platform::Place place_;
  std::string uid_slot_;

  // The input type of pipe reader, 0 for one sample, 1 for one batch
  int input_type_;
  int gpu_graph_mode_ = 0;
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
  GraphDataGenerator gpu_graph_data_generator_;
#endif
  bool train_mode_;
};

// PrivateQueueDataFeed is the base virtual class for ohther DataFeeds.
// It use a read-thread to read file and parse data to a private-queue
// (thread level), and get data from this queue when trainer call Next().
template <typename T>
class PrivateQueueDataFeed : public DataFeed {
 public:
  PrivateQueueDataFeed() {}
  virtual ~PrivateQueueDataFeed() {}
  virtual bool Start();
  virtual int Next();

 protected:
  // The thread implementation function for reading file and parse.
  virtual void ReadThread();
  // This function is used to set private-queue size, and the most
  // efficient when the queue size is close to the batch size.
  virtual void SetQueueSize(int queue_size);
  // The reading and parsing method called in the ReadThread.
  virtual bool ParseOneInstance(T* instance) = 0;
  virtual bool ParseOneInstanceFromPipe(T* instance) = 0;
  // This function is used to put instance to vec_ins
  virtual void AddInstanceToInsVec(T* vec_ins,
                                   const T& instance,
                                   int index) = 0;
  // This function is used to put ins_vec to feed_vec
  virtual void PutToFeedVec(const T& ins_vec) = 0;

  // The thread for read files
  std::thread read_thread_;
  // using ifstream one line and one line parse is faster
  // than using fread one buffer and one buffer parse.
  //   for a 601M real data:
  //     ifstream one line and one line parse: 6034 ms
  //     fread one buffer and one buffer parse: 7097 ms
  std::ifstream file_;
  std::shared_ptr<FILE> fp_;
  size_t queue_size_;
  string::LineFileReader reader_;
  // The queue for store parsed data
  std::shared_ptr<paddle::framework::ChannelObject<T>> queue_;
};

template <typename T>
class InMemoryDataFeed : public DataFeed {
 public:
  InMemoryDataFeed();
  virtual ~InMemoryDataFeed() {}
  virtual void Init(const DataFeedDesc& data_feed_desc) = 0;
  virtual bool Start();
  virtual int Next();
  virtual void SetInputPvChannel(void* channel);
  virtual void SetOutputPvChannel(void* channel);
  virtual void SetConsumePvChannel(void* channel);

  virtual void SetInputChannel(void* channel);
  virtual void SetOutputChannel(void* channel);
  virtual void SetConsumeChannel(void* channel);
  virtual void SetThreadId(int thread_id);
  virtual void SetThreadNum(int thread_num);
  virtual void SetParseInsId(bool parse_ins_id);
  virtual void SetParseUid(bool parse_uid);
  virtual void SetParseContent(bool parse_content);
  virtual void SetParseLogKey(bool parse_logkey);
  virtual void SetEnablePvMerge(bool enable_pv_merge);
  virtual void SetCurrentPhase(int current_phase);
  virtual void LoadIntoMemory();
  virtual void LoadIntoMemoryFromSo();
  virtual void SetRecord(T* records) { records_ = records; }
  int GetDefaultBatchSize() { return default_batch_size_; }
  void AddBatchOffset(const std::pair<int, int>& offset) {
    batch_offsets_.push_back(offset);
  }

 protected:
  virtual bool ParseOneInstance(T* instance) = 0;
  virtual bool ParseOneInstanceFromPipe(T* instance) = 0;
  virtual void ParseOneInstanceFromSo(const char* str,
                                      T* instance,
                                      CustomParser* parser) {}
  virtual int ParseInstanceFromSo(int len,
                                  const char* str,
                                  std::vector<T>* instances,
                                  CustomParser* parser) {
    return 0;
  }
  virtual void PutToFeedVec(const std::vector<T>& ins_vec) = 0;
  virtual void PutToFeedVec(const T* ins_vec, int num) = 0;

  std::vector<std::vector<float>> batch_float_feasigns_;
  std::vector<std::vector<uint64_t>> batch_uint64_feasigns_;
  std::vector<std::vector<size_t>> offset_;
  std::vector<bool> visit_;

  int thread_id_;
  int thread_num_;
  bool parse_ins_id_;
  bool parse_uid_;
  bool parse_content_;
  bool parse_logkey_;
  bool enable_pv_merge_;
  int current_phase_{-1};  // only for untest
  std::ifstream file_;
  std::shared_ptr<FILE> fp_;
  paddle::framework::ChannelObject<T>* input_channel_;
  paddle::framework::ChannelObject<T>* output_channel_;
  paddle::framework::ChannelObject<T>* consume_channel_;

  paddle::framework::ChannelObject<PvInstance>* input_pv_channel_;
  paddle::framework::ChannelObject<PvInstance>* output_pv_channel_;
  paddle::framework::ChannelObject<PvInstance>* consume_pv_channel_;

  std::vector<std::pair<int, int>> batch_offsets_;
  uint64_t offset_index_ = 0;
  bool enable_heterps_ = false;
  T* records_ = nullptr;
};

// This class define the data type of instance(ins_vec) in MultiSlotDataFeed
class MultiSlotType {
 public:
  MultiSlotType() {}
  ~MultiSlotType() {}
  void Init(const std::string& type, size_t reserved_size = 0) {
    CheckType(type);
    if (type_[0] == 'f') {
      float_feasign_.clear();
      if (reserved_size) {
        float_feasign_.reserve(reserved_size);
      }
    } else if (type_[0] == 'u') {
      uint64_feasign_.clear();
      if (reserved_size) {
        uint64_feasign_.reserve(reserved_size);
      }
    }
    type_ = type;
  }
  void InitOffset(size_t max_batch_size = 0) {
    if (max_batch_size > 0) {
      offset_.reserve(max_batch_size + 1);
    }
    offset_.resize(1);
    // LoDTensor' lod is counted from 0, the size of lod
    // is one size larger than the size of data.
    offset_[0] = 0;
  }
  const std::vector<size_t>& GetOffset() const { return offset_; }
  std::vector<size_t>& MutableOffset() { return offset_; }
  void AddValue(const float v) {
    CheckFloat();
    float_feasign_.push_back(v);
  }
  void AddValue(const uint64_t v) {
    CheckUint64();
    uint64_feasign_.push_back(v);
  }
  void CopyValues(const float* input, size_t size) {
    CheckFloat();
    float_feasign_.resize(size);
    memcpy(float_feasign_.data(), input, size * sizeof(float));
  }
  void CopyValues(const uint64_t* input, size_t size) {
    CheckUint64();
    uint64_feasign_.resize(size);
    memcpy(uint64_feasign_.data(), input, size * sizeof(uint64_t));
  }
  void AddIns(const MultiSlotType& ins) {
    if (ins.GetType()[0] == 'f') {  // float
      CheckFloat();
      auto& vec = ins.GetFloatData();
      offset_.push_back(offset_.back() + vec.size());
      float_feasign_.insert(float_feasign_.end(), vec.begin(), vec.end());
    } else if (ins.GetType()[0] == 'u') {  // uint64
      CheckUint64();
      auto& vec = ins.GetUint64Data();
      offset_.push_back(offset_.back() + vec.size());
      uint64_feasign_.insert(uint64_feasign_.end(), vec.begin(), vec.end());
    }
  }
  void AppendValues(const uint64_t* input, size_t size) {
    CheckUint64();
    offset_.push_back(offset_.back() + size);
    uint64_feasign_.insert(uint64_feasign_.end(), input, input + size);
  }
  void AppendValues(const float* input, size_t size) {
    CheckFloat();
    offset_.push_back(offset_.back() + size);

    float_feasign_.insert(float_feasign_.end(), input, input + size);
  }
  const std::vector<float>& GetFloatData() const { return float_feasign_; }
  std::vector<float>& MutableFloatData() { return float_feasign_; }
  const std::vector<uint64_t>& GetUint64Data() const { return uint64_feasign_; }
  std::vector<uint64_t>& MutableUint64Data() { return uint64_feasign_; }
  const std::string& GetType() const { return type_; }
  size_t GetBatchSize() { return offset_.size() - 1; }
  std::string& MutableType() { return type_; }

  std::string DebugString() {
    std::stringstream ss;

    ss << "\ntype: " << type_ << "\n";
    ss << "offset: ";
    ss << "[";
    for (const size_t& i : offset_) {
      ss << offset_[i] << ",";
    }
    ss << "]\ndata: [";
    if (type_[0] == 'f') {
      for (const float& i : float_feasign_) {
        ss << i << ",";
      }
    } else {
      for (const uint64_t& i : uint64_feasign_) {
        ss << i << ",";
      }
    }
    ss << "]\n";
    return ss.str();
  }

 private:
  void CheckType(const std::string& type) const {
    PADDLE_ENFORCE_EQ((type == "uint64" || type == "float"),
                      true,
                      platform::errors::InvalidArgument(
                          "MultiSlotType error, expect type is uint64 or "
                          "float, but received type is %s.",
                          type));
  }
  void CheckFloat() const {
    PADDLE_ENFORCE_EQ(
        type_[0],
        'f',
        platform::errors::InvalidArgument(
            "MultiSlotType error, add %s value to float slot.", type_));
  }
  void CheckUint64() const {
    PADDLE_ENFORCE_EQ(
        type_[0],
        'u',
        platform::errors::InvalidArgument(
            "MultiSlotType error, add %s value to uint64 slot.", type_));
  }
  std::vector<float> float_feasign_;
  std::vector<uint64_t> uint64_feasign_;
  std::string type_;
  std::vector<size_t> offset_;
};

template <class AR>
paddle::framework::Archive<AR>& operator<<(paddle::framework::Archive<AR>& ar,
                                           const MultiSlotType& ins) {
  ar << ins.GetType();
#ifdef _LINUX
  ar << ins.GetOffset();
#else
  const auto& offset = ins.GetOffset();
  ar << (uint64_t)offset.size();
  for (const size_t& x : offset) {
    ar << (const uint64_t)x;
  }
#endif
  ar << ins.GetFloatData();
  ar << ins.GetUint64Data();
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator>>(paddle::framework::Archive<AR>& ar,
                                           MultiSlotType& ins) {
  ar >> ins.MutableType();
#ifdef _LINUX
  ar >> ins.MutableOffset();
#else
  auto& offset = ins.MutableOffset();
  offset.resize(ar.template Get<uint64_t>());
  for (size_t& x : offset) {
    uint64_t t;
    ar >> t;
    x = static_cast<size_t>(t);
  }
#endif
  ar >> ins.MutableFloatData();
  ar >> ins.MutableUint64Data();
  return ar;
}

struct RecordCandidate {
  std::string ins_id_;
  std::unordered_multimap<uint16_t, FeatureFeasign> feas_;
  size_t shadow_index_ = -1;  // Optimization for Reservoir Sample

  RecordCandidate() {}
  RecordCandidate(const Record& rec,
                  const std::unordered_set<uint16_t>& slot_index_to_replace) {
    for (const auto& fea : rec.uint64_feasigns_) {
      if (slot_index_to_replace.find(fea.slot()) !=
          slot_index_to_replace.end()) {
        feas_.insert({fea.slot(), fea.sign()});
      }
    }
  }

  RecordCandidate& operator=(const Record& rec) {
    feas_.clear();
    ins_id_ = rec.ins_id_;
    for (auto& fea : rec.uint64_feasigns_) {
      feas_.insert({fea.slot(), fea.sign()});
    }
    return *this;
  }
};

class RecordCandidateList {
 public:
  RecordCandidateList() = default;
  RecordCandidateList(const RecordCandidateList&) {}

  size_t Size() { return cur_size_; }
  void ReSize(size_t length);

  void ReInit();
  void ReInitPass() {
    for (size_t i = 0; i < cur_size_; ++i) {
      if (candidate_list_[i].shadow_index_ != i) {
        candidate_list_[i].ins_id_ =
            candidate_list_[candidate_list_[i].shadow_index_].ins_id_;
        candidate_list_[i].feas_.swap(
            candidate_list_[candidate_list_[i].shadow_index_].feas_);
        candidate_list_[i].shadow_index_ = i;
      }
    }
    candidate_list_.resize(cur_size_);
  }

  void AddAndGet(const Record& record, RecordCandidate* result);
  void AddAndGet(const Record& record, size_t& index_result) {  // NOLINT
    // std::unique_lock<std::mutex> lock(mutex_);
    size_t index = 0;
    ++total_size_;
    auto fleet_ptr = FleetWrapper::GetInstance();
    if (!full_) {
      candidate_list_.emplace_back(record, slot_index_to_replace_);
      candidate_list_.back().shadow_index_ = cur_size_;
      ++cur_size_;
      full_ = (cur_size_ == capacity_);
    } else {
      index = fleet_ptr->LocalRandomEngine()() % total_size_;
      if (index < capacity_) {
        candidate_list_.emplace_back(record, slot_index_to_replace_);
        candidate_list_[index].shadow_index_ = candidate_list_.size() - 1;
      }
    }
    index = fleet_ptr->LocalRandomEngine()() % cur_size_;
    index_result = candidate_list_[index].shadow_index_;
  }
  const RecordCandidate& Get(size_t index) const {
    PADDLE_ENFORCE_LT(
        index,
        candidate_list_.size(),
        platform::errors::OutOfRange("Your index [%lu] exceeds the number of "
                                     "elements in candidate_list[%lu].",
                                     index,
                                     candidate_list_.size()));
    return candidate_list_[index];
  }
  void SetSlotIndexToReplace(
      const std::unordered_set<uint16_t>& slot_index_to_replace) {
    slot_index_to_replace_ = slot_index_to_replace;
  }

 private:
  size_t capacity_ = 0;
  std::mutex mutex_;
  bool full_ = false;
  size_t cur_size_ = 0;
  size_t total_size_ = 0;
  std::vector<RecordCandidate> candidate_list_;
  std::unordered_set<uint16_t> slot_index_to_replace_;
};

template <class AR>
paddle::framework::Archive<AR>& operator<<(paddle::framework::Archive<AR>& ar,
                                           const FeatureFeasign& fk) {
  ar << fk.uint64_feasign_;
  ar << fk.float_feasign_;
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator>>(paddle::framework::Archive<AR>& ar,
                                           FeatureFeasign& fk) {
  ar >> fk.uint64_feasign_;
  ar >> fk.float_feasign_;
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator<<(paddle::framework::Archive<AR>& ar,
                                           const FeatureItem& fi) {
  ar << fi.sign();
  ar << fi.slot();
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator>>(paddle::framework::Archive<AR>& ar,
                                           FeatureItem& fi) {
  ar >> fi.sign();
  ar >> fi.slot();
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator<<(paddle::framework::Archive<AR>& ar,
                                           const Record& r) {
  ar << r.uint64_feasigns_;
  ar << r.float_feasigns_;
  ar << r.ins_id_;
  return ar;
}

template <class AR>
paddle::framework::Archive<AR>& operator>>(paddle::framework::Archive<AR>& ar,
                                           Record& r) {
  ar >> r.uint64_feasigns_;
  ar >> r.float_feasigns_;
  ar >> r.ins_id_;
  return ar;
}

// This DataFeed is used to feed multi-slot type data.
// The format of multi-slot type data:
//   [n feasign_0 feasign_1 ... feasign_n]*
class MultiSlotDataFeed
    : public PrivateQueueDataFeed<std::vector<MultiSlotType>> {
 public:
  MultiSlotDataFeed() {}
  virtual ~MultiSlotDataFeed() {}
  virtual void Init(const DataFeedDesc& data_feed_desc);
  virtual bool CheckFile(const char* filename);

 protected:
  virtual void ReadThread();
  virtual void AddInstanceToInsVec(std::vector<MultiSlotType>* vec_ins,
                                   const std::vector<MultiSlotType>& instance,
                                   int index);
  virtual bool ParseOneInstance(std::vector<MultiSlotType>* instance);
  virtual bool ParseOneInstanceFromPipe(std::vector<MultiSlotType>* instance);
  virtual void PutToFeedVec(const std::vector<MultiSlotType>& ins_vec);
};

class MultiSlotInMemoryDataFeed : public InMemoryDataFeed<Record> {
 public:
  MultiSlotInMemoryDataFeed() {}
  virtual ~MultiSlotInMemoryDataFeed() {}
  virtual void Init(const DataFeedDesc& data_feed_desc);
  // void SetRecord(Record* records) { records_ = records; }

 protected:
  virtual bool ParseOneInstance(Record* instance);
  virtual bool ParseOneInstanceFromPipe(Record* instance);
  virtual void ParseOneInstanceFromSo(const char* str,
                                      Record* instance,
                                      CustomParser* parser){};
  virtual int ParseInstanceFromSo(int len,
                                  const char* str,
                                  std::vector<Record>* instances,
                                  CustomParser* parser);
  virtual void PutToFeedVec(const std::vector<Record>& ins_vec);
  virtual void GetMsgFromLogKey(const std::string& log_key,
                                uint64_t* search_id,
                                uint32_t* cmatch,
                                uint32_t* rank);
  virtual void PutToFeedVec(const Record* ins_vec, int num);
};

class SlotRecordInMemoryDataFeed : public InMemoryDataFeed<SlotRecord> {
 public:
  SlotRecordInMemoryDataFeed() {}
  virtual ~SlotRecordInMemoryDataFeed() {
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
    if (pack_ != nullptr) {
      pack_ = nullptr;
    }
#endif
  }
  virtual void Init(const DataFeedDesc& data_feed_desc);
  virtual void LoadIntoMemory();
  void ExpandSlotRecord(SlotRecord* ins);

 protected:
  virtual bool Start();
  virtual int Next();
  virtual bool ParseOneInstance(SlotRecord* instance) { return false; }
  virtual bool ParseOneInstanceFromPipe(SlotRecord* instance) { return false; }
  // virtual void ParseOneInstanceFromSo(const char* str, T* instance,
  //                                    CustomParser* parser) {}
  virtual void PutToFeedVec(const std::vector<SlotRecord>& ins_vec) {}

  virtual void LoadIntoMemoryByCommand(void);
  virtual void LoadIntoMemoryByLib(void);
  virtual void LoadIntoMemoryByLine(void);
  virtual void LoadIntoMemoryByFile(void);
  virtual void SetInputChannel(void* channel) {
    input_channel_ = static_cast<ChannelObject<SlotRecord>*>(channel);
  }
  bool ParseOneInstance(const std::string& line, SlotRecord* rec);
  virtual void PutToFeedVec(const SlotRecord* ins_vec, int num);
  virtual void AssignFeedVar(const Scope& scope);
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  void BuildSlotBatchGPU(const int ins_num);
  void FillSlotValueOffset(const int ins_num,
                           const int used_slot_num,
                           size_t* slot_value_offsets,
                           const int* uint64_offsets,
                           const int uint64_slot_size,
                           const int* float_offsets,
                           const int float_slot_size,
                           const UsedSlotGpuType* used_slots);
  void CopyForTensor(const int ins_num,
                     const int used_slot_num,
                     void** dest,
                     const size_t* slot_value_offsets,
                     const uint64_t* uint64_feas,
                     const int* uint64_offsets,
                     const int* uint64_ins_lens,
                     const int uint64_slot_size,
                     const float* float_feas,
                     const int* float_offsets,
                     const int* float_ins_lens,
                     const int float_slot_size,
                     const UsedSlotGpuType* used_slots);
#endif
  float sample_rate_ = 1.0f;
  int use_slot_size_ = 0;
  int float_use_slot_size_ = 0;
  int uint64_use_slot_size_ = 0;
  std::vector<AllSlotInfo> all_slots_info_;
  std::vector<UsedSlotInfo> used_slots_info_;
  size_t float_total_dims_size_ = 0;
  std::vector<int> float_total_dims_without_inductives_;

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  MiniBatchGpuPack* pack_ = nullptr;
#endif
};

class PaddleBoxDataFeed : public MultiSlotInMemoryDataFeed {
 public:
  PaddleBoxDataFeed() {}
  virtual ~PaddleBoxDataFeed() {}

 protected:
  virtual void Init(const DataFeedDesc& data_feed_desc);
  virtual bool Start();
  virtual int Next();
  virtual void AssignFeedVar(const Scope& scope);
  virtual void PutToFeedVec(const std::vector<PvInstance>& pv_vec);
  virtual void PutToFeedVec(const std::vector<Record*>& ins_vec);
  virtual int GetCurrentPhase();
  virtual void GetRankOffset(const std::vector<PvInstance>& pv_vec,
                             int ins_number);
  std::string rank_offset_name_;
  int pv_batch_size_;
};

#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && !defined(_WIN32)
template <typename T>
class PrivateInstantDataFeed : public DataFeed {
 public:
  PrivateInstantDataFeed() {}
  virtual ~PrivateInstantDataFeed() {}
  void Init(const DataFeedDesc& data_feed_desc) override;
  bool Start() override { return true; }
  int Next() override;

 protected:
  // The batched data buffer
  std::vector<MultiSlotType> ins_vec_;

  // This function is used to preprocess with a given filename, e.g. open it or
  // mmap
  virtual bool Preprocess(const std::string& filename) = 0;

  // This function is used to postprocess system resource such as closing file
  // NOTICE: Ensure that it is safe to call before Preprocess
  virtual bool Postprocess() = 0;

  // The reading and parsing method.
  virtual bool ParseOneMiniBatch() = 0;

  // This function is used to put ins_vec to feed_vec
  virtual void PutToFeedVec();
};

class MultiSlotFileInstantDataFeed
    : public PrivateInstantDataFeed<std::vector<MultiSlotType>> {
 public:
  MultiSlotFileInstantDataFeed() {}
  virtual ~MultiSlotFileInstantDataFeed() {}

 protected:
  int fd_{-1};
  char* buffer_{nullptr};
  size_t end_{0};
  size_t offset_{0};

  bool Preprocess(const std::string& filename) override;

  bool Postprocess() override;

  bool ParseOneMiniBatch() override;
};
#endif

}  // namespace framework
}  // namespace paddle
