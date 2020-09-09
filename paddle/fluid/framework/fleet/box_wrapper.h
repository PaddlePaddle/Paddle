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
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_extends.h>
#include <boxps_public.h>
#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif
#include <glog/logging.h>
#include <algorithm>
#include <atomic>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/string/string_helper.h"
#define BUF_SIZE 1024 * 1024

DECLARE_int32(fix_dayid);
namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_BOX_PS
class BasicAucCalculator {
 public:
  explicit BasicAucCalculator(bool mode_collect_in_gpu = false)
      : _mode_collect_in_gpu(mode_collect_in_gpu) {}
  void init(int table_size, int max_batch_size = 0);
  void reset();
  // add single data in CPU with LOCK, deprecated
  void add_unlock_data(double pred, int label);
  // add batch data
  void add_data(const float* d_pred, const int64_t* d_label, int batch_size,
                const paddle::platform::Place& place);
  // add mask data
  void add_mask_data(const float* d_pred, const int64_t* d_label,
                     const int64_t* d_mask, int batch_size,
                     const paddle::platform::Place& place);
  void compute();
  int table_size() const { return _table_size; }
  double bucket_error() const { return _bucket_error; }
  double auc() const { return _auc; }
  double mae() const { return _mae; }
  double actual_ctr() const { return _actual_ctr; }
  double predicted_ctr() const { return _predicted_ctr; }
  double size() const { return _size; }
  double rmse() const { return _rmse; }
  std::vector<double>& get_negative() { return _table[0]; }
  std::vector<double>& get_postive() { return _table[1]; }
  double& local_abserr() { return _local_abserr; }
  double& local_sqrerr() { return _local_sqrerr; }
  double& local_pred() { return _local_pred; }
  // lock and unlock
  std::mutex& table_mutex(void) { return _table_mutex; }

 private:
  void cuda_add_data(const paddle::platform::Place& place, const int64_t* label,
                     const float* pred, int len);
  void cuda_add_mask_data(const paddle::platform::Place& place,
                          const int64_t* label, const float* pred,
                          const int64_t* mask, int len);
  void calculate_bucket_error();

 protected:
  double _local_abserr = 0;
  double _local_sqrerr = 0;
  double _local_pred = 0;
  double _auc = 0;
  double _mae = 0;
  double _rmse = 0;
  double _actual_ctr = 0;
  double _predicted_ctr = 0;
  double _size;
  double _bucket_error = 0;

  std::vector<std::shared_ptr<memory::Allocation>> _d_positive;
  std::vector<std::shared_ptr<memory::Allocation>> _d_negative;
  std::vector<std::shared_ptr<memory::Allocation>> _d_abserr;
  std::vector<std::shared_ptr<memory::Allocation>> _d_sqrerr;
  std::vector<std::shared_ptr<memory::Allocation>> _d_pred;

 private:
  void set_table_size(int table_size) { _table_size = table_size; }
  void set_max_batch_size(int max_batch_size) {
    _max_batch_size = max_batch_size;
  }
  void collect_data_nccl();
  void copy_data_d2h(int device);
  int _table_size;
  int _max_batch_size;
  bool _mode_collect_in_gpu;
  std::vector<double> _table[2];
  static constexpr double kRelativeErrorBound = 0.05;
  static constexpr double kMaxSpan = 0.01;
  std::mutex _table_mutex;
};

class QueryEmbSet {
public:
  QueryEmbSet(int dim) {
    emb_dim = dim;
  }

  ~QueryEmbSet() {
    for (size_t i = 0; i < d_embs.size(); ++i) {
      cudaFree(d_embs[i]);
    }
  }
  int AddEmb(std::vector<float>& emb) {
    int r;
    h_emb_mtx.lock();
    h_emb.insert(h_emb.end(), emb.begin(), emb.end());
    r = h_emb_count;
    ++h_emb_count;
    h_emb_mtx.unlock();
    return r;
  }

  void to_hbm() {
    for (int i = 0; i < 8; ++i) {
      d_embs.push_back(NULL);
      cudaSetDevice(i);
      cudaMalloc(&d_embs.back(), h_emb_count * emb_dim * sizeof(float));
      auto place = platform::CUDAPlace(i);
      auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
      cudaMemcpyAsync(d_embs.back(), h_emb.data(), h_emb_count * emb_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
  }

  void PullQueryEmb(uint64_t* d_keys, float* d_vals, int num, int gpu_id);

  int emb_dim=0;
  int h_emb_count=0;
  std::mutex h_emb_mtx;
  std::vector<float> h_emb;
  std::vector<float*> d_embs;

};


class BoxWrapper {
  struct DeviceBoxData {
    LoDTensor keys_tensor;
    LoDTensor dims_tensor;
    std::shared_ptr<memory::Allocation> pull_push_buf = nullptr;
    std::shared_ptr<memory::Allocation> gpu_keys_ptr = nullptr;
    std::shared_ptr<memory::Allocation> gpu_values_ptr = nullptr;

    LoDTensor slot_lens;
    LoDTensor d_slot_vector;
    LoDTensor keys2slot;

    platform::Timer all_pull_timer;
    platform::Timer boxps_pull_timer;
    platform::Timer all_push_timer;
    platform::Timer boxps_push_timer;

    int64_t total_key_length = 0;

    void ResetTimer(void) {
      all_pull_timer.Reset();
      boxps_pull_timer.Reset();
      all_push_timer.Reset();
      boxps_push_timer.Reset();
    }
  };

 public:
  std::deque<QueryEmbSet> query_emb_set_q;
  virtual ~BoxWrapper() {
    if (file_manager_ != nullptr) {
      file_manager_->destory();
      file_manager_ = nullptr;
    }
    if (data_shuffle_ != nullptr) {
      data_shuffle_->destory();
      data_shuffle_ = nullptr;
    }
    if (p_agent_ != nullptr) {
      delete p_agent_;
      p_agent_ = nullptr;
    }
    if (device_caches_ != nullptr) {
      delete device_caches_;
      device_caches_ = nullptr;
    }
  }
  BoxWrapper() {
    fprintf(stdout, "init box wrapper\n");
    boxps::MPICluster::Ins();
  }

  void FeedPass(int date, const std::vector<uint64_t>& feasgin_to_box) ;
  void BeginFeedPass(int date, boxps::PSAgentBase** agent) ;
  void EndFeedPass(boxps::PSAgentBase* agent) ;
  void BeginPass() ;
  void EndPass(bool need_save_delta) ;
  void SetTestMode(bool is_test) const;

  template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM = 0>
  void PullSparseCase(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<float*>& values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size, const int expand_embed_dim);

  void PullSparse(const paddle::platform::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size, const int expand_embed_dim);

  template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM = 0>
  void PushSparseGradCase(const paddle::platform::Place& place,
                          const std::vector<const uint64_t*>& keys,
                          const std::vector<const float*>& grad_values,
                          const std::vector<int64_t>& slot_lengths,
                          const int hidden_size, const int expand_embed_dim,
                          const int batch_size);

  void PushSparseGrad(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size, const int expand_embed_dim,
                      const int batch_size);

  void CopyForPull(const paddle::platform::Place& place, uint64_t** gpu_keys,
                   float** gpu_values, void* total_values_gpu,
                   const int64_t* slot_lens, const int slot_num,
                   const int* key2slot, const int hidden_size,
                   const int expand_embed_dim, const int64_t total_length,
                   int* total_dims);

  void CopyForPush(const paddle::platform::Place& place, float** grad_values,
                   void* total_grad_values_gpu, const int* slots,
                   const int64_t* slot_lens, const int slot_num,
                   const int hidden_size, const int expand_embed_dim,
                   const int64_t total_length, const int batch_size,
                   const int* total_dims, const int* key2slot);

  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len, int* key2slot);

  void CheckEmbedSizeIsValid(int embedx_dim, int expand_embed_dim);

  boxps::PSAgentBase* GetAgent() { return p_agent_; }
  void InitializeGPUAndLoadModel(
      const char* conf_file, const std::vector<int>& slot_vector,
      const std::vector<std::string>& slot_omit_in_feedpass,
      const std::string& model_path, const std::map<std::string, float> &lr_map, bool is_hbm_query) {
    if (nullptr != s_instance_) {
      VLOG(3) << "Begin InitializeGPU";
      std::vector<cudaStream_t*> stream_list;
      int gpu_num = platform::GetCUDADeviceCount();
      for (int i = 0; i < gpu_num; ++i) {
        VLOG(3) << "before get context i[" << i << "]";
        platform::CUDADeviceContext* context =
            dynamic_cast<platform::CUDADeviceContext*>(
                platform::DeviceContextPool::Instance().Get(
                    platform::CUDAPlace(i)));
        stream_list_[i] = context->stream();
        stream_list.push_back(&stream_list_[i]);
      }
      VLOG(2) << "Begin call InitializeGPU in BoxPS";
      // the second parameter is useless
      s_instance_->boxps_ptr_->InitializeGPUAndLoadModel(
          conf_file, -1, stream_list, slot_vector, model_path);
      p_agent_ = boxps::PSAgentBase::GetIns(feedpass_thread_num_);
      p_agent_->Init();
      for (const auto& slot_name : slot_omit_in_feedpass) {
        slot_name_omited_in_feedpass_.insert(slot_name);
      }
      is_hbm_query_ = is_hbm_query;
      slot_vector_ = slot_vector;
      device_caches_ = new DeviceBoxData[gpu_num];

      VLOG(0) << "lr_map.size(): " << lr_map.size();
      for (const auto e: lr_map) {
        VLOG(0) << e.first << "'s lr is " << e.second;
        if (e.first.find("param") != std::string::npos) {
          lr_map_[e.first + ".w_0"] = e.second;
          lr_map_[e.first + ".b_0"] = e.second;
        }
      }
    }
  }

  int GetFeedpassThreadNum() const { return feedpass_thread_num_; }

  void Finalize() {
    VLOG(3) << "Begin Finalize";
    if (nullptr != s_instance_ && s_instance_->boxps_ptr_ != nullptr) {
      s_instance_->boxps_ptr_->Finalize();
      s_instance_->boxps_ptr_ = nullptr;
    }
  }

  void ReleasePool(void) {
    // after one day train release memory pool slot record
    platform::Timer timer;
    timer.Start();
    size_t capacity = SlotRecordPool().capacity();
    SlotRecordPool().clear();
    timer.Pause();
    STAT_RESET(STAT_total_feasign_num_in_mem, 0);
    STAT_RESET(STAT_slot_pool_size, 0);
    LOG(WARNING) << "ReleasePool Size=" << capacity
                 << ", Time=" << timer.ElapsedSec() << "sec";
  }

  const std::string SaveBase(const char* batch_model_path,
                             const char* xbox_model_path,
                             const std::string& date) {
    VLOG(3) << "Begin SaveBase";
    PADDLE_ENFORCE_EQ(
        date.length(), 8,
        platform::errors::PreconditionNotMet(
            "date[%s] is invalid, correct example is 20190817", date.c_str()));
    int year = std::stoi(date.substr(0, 4));
    int month = std::stoi(date.substr(4, 2));
    int day = std::stoi(date.substr(6, 2));

    struct std::tm b;
    b.tm_year = year - 1900;
    b.tm_mon = month - 1;
    b.tm_mday = day;
    b.tm_hour = FLAGS_fix_dayid ? 8 : 0;
    b.tm_min = b.tm_sec = 0;
    std::time_t seconds_from_1970 = std::mktime(&b);

    std::string ret_str;
    int ret = boxps_ptr_->SaveBase(batch_model_path, xbox_model_path, ret_str,
                                   seconds_from_1970 / 86400);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "SaveBase failed in BoxPS."));
    return ret_str;
  }

  const std::string SaveDelta(const char* xbox_model_path) {
    VLOG(3) << "Begin SaveDelta";
    std::string ret_str;
    int ret = boxps_ptr_->SaveDelta(xbox_model_path, ret_str);
    PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                  "SaveDelta failed in BoxPS."));
    return ret_str;
  }

  static std::shared_ptr<BoxWrapper> GetInstance() {
    PADDLE_ENFORCE_EQ(
        s_instance_ == nullptr, false,
        platform::errors::PreconditionNotMet(
            "GetInstance failed in BoxPs, you should use SetInstance firstly"));
    return s_instance_;
  }

  static std::shared_ptr<BoxWrapper> SetInstance(int embedx_dim = 8,
                                                 int expand_embed_dim = 0) {
    if (nullptr == s_instance_) {
      // If main thread is guaranteed to init this, this lock can be removed
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (nullptr == s_instance_) {
        VLOG(3) << "s_instance_ is null";
        s_instance_.reset(new paddle::framework::BoxWrapper());
        s_instance_->boxps_ptr_.reset(
            boxps::BoxPSBase::GetIns(embedx_dim, expand_embed_dim));
        embedx_dim_ = embedx_dim;
        expand_embed_dim_ = expand_embed_dim;

        if (boxps::MPICluster::Ins().size() > 1) {
          data_shuffle_.reset(boxps::PaddleShuffler::New());
          data_shuffle_->init(10);
        }
      }
    } else {
      LOG(WARNING) << "You have already used SetInstance() before";
    }
    return s_instance_;
  }

  void InitAfsAPI(const std::string& fs_name, const std::string& fs_ugi,
                  const std::string& conf_path) {
    file_manager_.reset(boxps::PaddleFileMgr::New());
    auto split = fs_ugi.find(",");
    std::string user = fs_ugi.substr(0, split);
    std::string pwd = fs_ugi.substr(split + 1);
    bool ret = file_manager_->initialize(fs_name, user, pwd, conf_path);
    PADDLE_ENFORCE_EQ(ret, true, platform::errors::PreconditionNotMet(
                                     "Called AFSAPI Init Interface Failed."));
    use_afs_api_ = true;
  }

  bool UseAfsApi() const { return use_afs_api_; }

  std::shared_ptr<FILE> OpenReadFile(const std::string& path,
                                     const std::string& pipe_command) {
    return boxps::fopen_read(file_manager_.get(), path, pipe_command);
  }

  boxps::PaddleFileMgr* GetFileMgr(void) { return file_manager_.get(); }

  // this performs better than rand_r, especially large data
  static std::default_random_engine& LocalRandomEngine() {
    struct engine_wrapper_t {
      std::default_random_engine engine;
      engine_wrapper_t() {
        struct timespec tp;
        clock_gettime(CLOCK_REALTIME, &tp);
        double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
        static std::atomic<uint64_t> x(0);
        std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
        engine.seed(sseq);
      }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
  }

  const std::unordered_set<std::string>& GetOmitedSlot() const {
    return slot_name_omited_in_feedpass_;
  }

  class MetricMsg {
   public:
    MetricMsg() {}
    MetricMsg(const std::string& label_varname, const std::string& pred_varname,
              int metric_phase, int bucket_size = 1000000,
              bool mode_collect_in_gpu = false, int max_batch_size = 0)
        : label_varname_(label_varname),
          pred_varname_(pred_varname),
          metric_phase_(metric_phase) {
      calculator = new BasicAucCalculator(mode_collect_in_gpu);
      calculator->init(bucket_size, max_batch_size);
    }
    virtual ~MetricMsg() {}

    int MetricPhase() const { return metric_phase_; }
    BasicAucCalculator* GetCalculator() { return calculator; }
    virtual void add_data(const Scope* exe_scope,
                          const paddle::platform::Place& place) {
      int label_len = 0;
      const int64_t* label_data = NULL;
      int pred_len = 0;
      const float* pred_data = NULL;
      get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);
      get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);
      PADDLE_ENFORCE_EQ(label_len, pred_len,
                        platform::errors::PreconditionNotMet(
                            "the predict data length should be consistent with "
                            "the label data length"));
      calculator->add_data(pred_data, label_data, label_len, place);
    }
    template <class T = float>
    static void get_data(const Scope* exe_scope, const std::string& varname,
                         const T** data, int* len) {
      auto* var = exe_scope->FindVar(varname.c_str());
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound(
                   "Error: var %s is not found in scope.", varname.c_str()));
      auto& gpu_tensor = var->Get<LoDTensor>();
      *data = gpu_tensor.data<T>();
      *len = gpu_tensor.numel();
    }
    template <class T = float>
    static void get_data(const Scope* exe_scope, const std::string& varname,
                         std::vector<T>* data) {
      auto* var = exe_scope->FindVar(varname.c_str());
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound(
                   "Error: var %s is not found in scope.", varname.c_str()));
      auto& gpu_tensor = var->Get<LoDTensor>();
      auto* gpu_data = gpu_tensor.data<T>();
      auto len = gpu_tensor.numel();
      data->resize(len);
      cudaMemcpy(data->data(), gpu_data, sizeof(T) * len,
                 cudaMemcpyDeviceToHost);
    }
    static inline std::pair<int, int> parse_cmatch_rank(uint64_t x) {
      // first 32 bit store cmatch and second 32 bit store rank
      return std::make_pair(static_cast<int>(x >> 32),
                            static_cast<int>(x & 0xff));
    }

   protected:
    std::string label_varname_;
    std::string pred_varname_;
    int metric_phase_;
    BasicAucCalculator* calculator;
  };

  class MultiTaskMetricMsg : public MetricMsg {
   public:
    MultiTaskMetricMsg(const std::string& label_varname,
                       const std::string& pred_varname_list, int metric_phase,
                       const std::string& cmatch_rank_group,
                       const std::string& cmatch_rank_varname,
                       int bucket_size = 1000000) {
      label_varname_ = label_varname;
      cmatch_rank_varname_ = cmatch_rank_varname;
      metric_phase_ = metric_phase;
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
      for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
        const std::vector<std::string>& cur_cmatch_rank =
            string::split_string(cmatch_rank, "_");
        PADDLE_ENFORCE_EQ(
            cur_cmatch_rank.size(), 2,
            platform::errors::PreconditionNotMet(
                "illegal multitask auc spec: %s", cmatch_rank.c_str()));
        cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                   atoi(cur_cmatch_rank[1].c_str()));
      }
      for (const auto& pred_varname : string::split_string(pred_varname_list)) {
        pred_v.emplace_back(pred_varname);
      }
      PADDLE_ENFORCE_EQ(cmatch_rank_v.size(), pred_v.size(),
                        platform::errors::PreconditionNotMet(
                            "cmatch_rank's size [%lu] should be equal to pred "
                            "list's size [%lu], but ther are not equal",
                            cmatch_rank_v.size(), pred_v.size()));
    }
    virtual ~MultiTaskMetricMsg() {}
    void add_data(const Scope* exe_scope,
                  const paddle::platform::Place& place) override {
      std::vector<int64_t> cmatch_rank_data;
      get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
      std::vector<int64_t> label_data;
      get_data<int64_t>(exe_scope, label_varname_, &label_data);
      size_t batch_size = cmatch_rank_data.size();
      PADDLE_ENFORCE_EQ(
          batch_size, label_data.size(),
          platform::errors::PreconditionNotMet(
              "illegal batch size: batch_size[%lu] and label_data[%lu]",
              batch_size, label_data.size()));

      std::vector<std::vector<float>> pred_data_list(pred_v.size());
      for (size_t i = 0; i < pred_v.size(); ++i) {
        get_data<float>(exe_scope, pred_v[i], &pred_data_list[i]);
      }
      for (size_t i = 0; i < pred_data_list.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            batch_size, pred_data_list[i].size(),
            platform::errors::PreconditionNotMet(
                "illegal batch size: batch_size[%lu] and pred_data[%lu]",
                batch_size, pred_data_list[i].size()));
      }
      auto cal = GetCalculator();
      std::lock_guard<std::mutex> lock(cal->table_mutex());
      for (size_t i = 0; i < batch_size; ++i) {
        auto cmatch_rank_it =
            std::find(cmatch_rank_v.begin(), cmatch_rank_v.end(),
                      parse_cmatch_rank(cmatch_rank_data[i]));
        if (cmatch_rank_it != cmatch_rank_v.end()) {
          cal->add_unlock_data(pred_data_list[std::distance(
                                   cmatch_rank_v.begin(), cmatch_rank_it)][i],
                               label_data[i]);
        }
      }
    }

   protected:
    std::vector<std::pair<int, int>> cmatch_rank_v;
    std::vector<std::string> pred_v;
    std::string cmatch_rank_varname_;
  };
  class CmatchRankMetricMsg : public MetricMsg {
   public:
    CmatchRankMetricMsg(const std::string& label_varname,
                        const std::string& pred_varname, int metric_phase,
                        const std::string& cmatch_rank_group,
                        const std::string& cmatch_rank_varname,
                        bool ignore_rank = false, int bucket_size = 1000000) {
      label_varname_ = label_varname;
      pred_varname_ = pred_varname;
      cmatch_rank_varname_ = cmatch_rank_varname;
      metric_phase_ = metric_phase;
      ignore_rank_ = ignore_rank;
      calculator = new BasicAucCalculator();
      calculator->init(bucket_size);
      for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
        if (ignore_rank) {  // CmatchAUC
          cmatch_rank_v.emplace_back(atoi(cmatch_rank.c_str()), 0);
          continue;
        }
        const std::vector<std::string>& cur_cmatch_rank =
            string::split_string(cmatch_rank, "_");
        PADDLE_ENFORCE_EQ(
            cur_cmatch_rank.size(), 2,
            platform::errors::PreconditionNotMet(
                "illegal cmatch_rank auc spec: %s", cmatch_rank.c_str()));
        cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                   atoi(cur_cmatch_rank[1].c_str()));
      }
    }
    virtual ~CmatchRankMetricMsg() {}
    void add_data(const Scope* exe_scope,
                  const paddle::platform::Place& place) override {
      std::vector<int64_t> cmatch_rank_data;
      get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
      std::vector<int64_t> label_data;
      get_data<int64_t>(exe_scope, label_varname_, &label_data);
      std::vector<float> pred_data;
      get_data<float>(exe_scope, pred_varname_, &pred_data);
      size_t batch_size = cmatch_rank_data.size();
      PADDLE_ENFORCE_EQ(
          batch_size, label_data.size(),
          platform::errors::PreconditionNotMet(
              "illegal batch size: cmatch_rank[%lu] and label_data[%lu]",
              batch_size, label_data.size()));
      PADDLE_ENFORCE_EQ(
          batch_size, pred_data.size(),
          platform::errors::PreconditionNotMet(
              "illegal batch size: cmatch_rank[%lu] and pred_data[%lu]",
              batch_size, pred_data.size()));
      auto cal = GetCalculator();
      std::lock_guard<std::mutex> lock(cal->table_mutex());
      for (size_t i = 0; i < batch_size; ++i) {
        const auto& cur_cmatch_rank = parse_cmatch_rank(cmatch_rank_data[i]);
        for (size_t j = 0; j < cmatch_rank_v.size(); ++j) {
          bool is_matched = false;
          if (ignore_rank_) {
            is_matched = cmatch_rank_v[j].first == cur_cmatch_rank.first;
          } else {
            is_matched = cmatch_rank_v[j] == cur_cmatch_rank;
          }
          if (is_matched) {
            cal->add_unlock_data(pred_data[i], label_data[i]);
            break;
          }
        }
      }
    }

   protected:
    std::vector<std::pair<int, int>> cmatch_rank_v;
    std::string cmatch_rank_varname_;
    bool ignore_rank_;
  };
  class MaskMetricMsg : public MetricMsg {
   public:
    MaskMetricMsg(const std::string& label_varname,
                  const std::string& pred_varname, int metric_phase,
                  const std::string& mask_varname, int bucket_size = 1000000,
                  bool mode_collect_in_gpu = false, int max_batch_size = 0) {
      label_varname_ = label_varname;
      pred_varname_ = pred_varname;
      mask_varname_ = mask_varname;
      metric_phase_ = metric_phase;
      calculator = new BasicAucCalculator(mode_collect_in_gpu);
      calculator->init(bucket_size, max_batch_size);
    }
    virtual ~MaskMetricMsg() {}
    void add_data(const Scope* exe_scope,
                  const paddle::platform::Place& place) override {
      int label_len = 0;
      const int64_t* label_data = NULL;
      get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);

      int pred_len = 0;
      const float* pred_data = NULL;
      get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);

      int mask_len = 0;
      const int64_t* mask_data = NULL;
      get_data<int64_t>(exe_scope, mask_varname_, &mask_data, &mask_len);
      PADDLE_ENFORCE_EQ(label_len, mask_len,
                        platform::errors::PreconditionNotMet(
                            "the predict data length should be consistent with "
                            "the label data length"));
      auto cal = GetCalculator();
      cal->add_mask_data(pred_data, label_data, mask_data, label_len, place);
    }

   protected:
    std::string mask_varname_;
  };
  const std::vector<std::string> GetMetricNameList(
      int metric_phase = -1) const {
    VLOG(0) << "Want to Get metric phase: " << metric_phase;
    if (metric_phase == -1) {
      return metric_name_list_;
    } else {
      std::vector<std::string> ret;
      for (const auto& name : metric_name_list_) {
        const auto iter = metric_lists_.find(name);
        PADDLE_ENFORCE_NE(
            iter, metric_lists_.end(),
            platform::errors::InvalidArgument(
                "The metric name you provided is not registered."));

        if (iter->second->MetricPhase() == metric_phase) {
          VLOG(0) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", we want";
          ret.push_back(name);
        } else {
          VLOG(0) << name << "'s phase is " << iter->second->MetricPhase()
                  << ", not we want";
        }
      }
      return ret;
    }
  }
  int Phase() const { return phase_; }
  void FlipPhase() { phase_ = (phase_ + 1) % phase_num_; }
  const std::map<std::string, float> GetLRMap() const { return lr_map_; }
  std::map<std::string, MetricMsg*>& GetMetricList() { return metric_lists_; }

  void InitMetric(const std::string& method, const std::string& name,
                  const std::string& label_varname,
                  const std::string& pred_varname,
                  const std::string& cmatch_rank_varname,
                  const std::string& mask_varname, int metric_phase,
                  const std::string& cmatch_rank_group, bool ignore_rank,
                  int bucket_size = 1000000, bool mode_collect_in_gpu = false,
                  int max_batch_size = 0) {
    if (method == "AucCalculator") {
      metric_lists_.emplace(
          name,
          new MetricMsg(label_varname, pred_varname, metric_phase, bucket_size,
                        mode_collect_in_gpu, max_batch_size));
    } else if (method == "MultiTaskAucCalculator") {
      metric_lists_.emplace(
          name, new MultiTaskMetricMsg(label_varname, pred_varname,
                                       metric_phase, cmatch_rank_group,
                                       cmatch_rank_varname, bucket_size));
    } else if (method == "CmatchRankAucCalculator") {
      metric_lists_.emplace(name, new CmatchRankMetricMsg(
                                      label_varname, pred_varname, metric_phase,
                                      cmatch_rank_group, cmatch_rank_varname,
                                      ignore_rank, bucket_size));
    } else if (method == "MaskAucCalculator") {
      metric_lists_.emplace(
          name, new MaskMetricMsg(label_varname, pred_varname, metric_phase,
                                  mask_varname, bucket_size,
                                  mode_collect_in_gpu, max_batch_size));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "PaddleBox only support AucCalculator, MultiTaskAucCalculator "
          "CmatchRankAucCalculator and MaskAucCalculator"));
    }
    metric_name_list_.emplace_back(name);
  }

  const std::vector<float> GetMetricMsg(const std::string& name) {
    const auto iter = metric_lists_.find(name);
    PADDLE_ENFORCE_NE(iter, metric_lists_.end(),
                      platform::errors::InvalidArgument(
                          "The metric name you provided is not registered."));
    std::vector<float> metric_return_values_(8, 0.0);
    auto* auc_cal_ = iter->second->GetCalculator();
    auc_cal_->compute();
    metric_return_values_[0] = auc_cal_->auc();
    metric_return_values_[1] = auc_cal_->bucket_error();
    metric_return_values_[2] = auc_cal_->mae();
    metric_return_values_[3] = auc_cal_->rmse();
    metric_return_values_[4] = auc_cal_->actual_ctr();
    metric_return_values_[5] = auc_cal_->predicted_ctr();
    metric_return_values_[6] =
        auc_cal_->actual_ctr() / auc_cal_->predicted_ctr();
    metric_return_values_[7] = auc_cal_->size();
    auc_cal_->reset();
    return metric_return_values_;
  }

 private:
  static cudaStream_t stream_list_[8];
  std::shared_ptr<boxps::BoxPSBase> boxps_ptr_ = nullptr;
  boxps::PSAgentBase* p_agent_ = nullptr;
  // TODO(hutuxian): magic number, will add a config to specify
  const int feedpass_thread_num_ = 30;  // magic number
  static std::shared_ptr<BoxWrapper> s_instance_;
  std::unordered_set<std::string> slot_name_omited_in_feedpass_;
  // EMBEDX_DIM and EXPAND_EMBED_DIM
  static int embedx_dim_;
  static int expand_embed_dim_;

  // Metric Related
  int phase_ = 1;
  int phase_num_ = 2;
  std::map<std::string, MetricMsg*> metric_lists_;
  std::vector<std::string> metric_name_list_;
  std::vector<int> slot_vector_;
  bool use_afs_api_ = false;
  std::shared_ptr<boxps::PaddleFileMgr> file_manager_ = nullptr;
  // box device cache
  DeviceBoxData* device_caches_ = nullptr;
  std::map<std::string, float> lr_map_;
  bool is_hbm_query_ = false;
 public:
  static std::shared_ptr<boxps::PaddleShuffler> data_shuffle_;

  // Auc Runner
 public:
  void InitializeAucRunner(std::vector<std::vector<std::string>> slot_eval,
                           int thread_num, int pool_size,
                           std::vector<std::string> slot_list) {
    mode_ = 1;
    phase_num_ = static_cast<int>(slot_eval.size());
    phase_ = phase_num_ - 1;
    auc_runner_thread_num_ = thread_num;
    pass_done_semi_ = paddle::framework::MakeChannel<int>();
    pass_done_semi_->Put(1);  // Note: At most 1 pipeline in AucRunner
    random_ins_pool_list.resize(thread_num);

    std::unordered_set<std::string> slot_set;
    for (size_t i = 0; i < slot_eval.size(); ++i) {
      for (const auto& slot : slot_eval[i]) {
        slot_set.insert(slot);
      }
    }
    for (size_t i = 0; i < slot_list.size(); ++i) {
      if (slot_set.find(slot_list[i]) != slot_set.end()) {
        slot_index_to_replace_.insert(static_cast<int16_t>(i));
      }
    }
    for (int i = 0; i < auc_runner_thread_num_; ++i) {
      random_ins_pool_list[i].SetSlotIndexToReplace(slot_index_to_replace_);
    }
    VLOG(0) << "AucRunner configuration: thread number[" << thread_num
            << "], pool size[" << pool_size << "], runner_group[" << phase_num_
            << "]";
    VLOG(0) << "Slots that need to be evaluated:";
    for (auto e : slot_index_to_replace_) {
      VLOG(0) << e << ": " << slot_list[e];
    }
  }
  void GetRandomReplace(const std::vector<Record>& pass_data);
  void AddReplaceFeasign(boxps::PSAgentBase* p_agent, int feed_pass_thread_num);
  void GetRandomData(const std::vector<Record>& pass_data,
                     const std::unordered_set<uint16_t>& slots_to_replace,
                     std::vector<Record>* result);
  int Mode() const { return mode_; }

 private:
  int mode_ = 0;  // 0 means train/test 1 means auc_runner
  int auc_runner_thread_num_ = 1;
  bool init_done_ = false;
  paddle::framework::Channel<int> pass_done_semi_;
  std::unordered_set<uint16_t> slot_index_to_replace_;
  std::vector<RecordCandidateList> random_ins_pool_list;
  std::vector<size_t> replace_idx_;
};
#endif

class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void SetDate(int year, int month, int day) {
    year_ = year;
    month_ = month;
    day_ = day;
  }
  void BeginPass() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
#endif
  }
  void EndPass(bool need_save_delta) {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass(need_save_delta);
#endif
  }

  void ReadData2Memory() {
    platform::Timer timer;
    VLOG(3) << "Begin ReadData2Memory(), dataset[" << dataset_ << "]";
    double feed_pass_span = 0.0;
    double read_ins_span = 0.0;

    timer.Start();
#ifdef PADDLE_WITH_BOX_PS
    struct std::tm b;
    b.tm_year = year_ - 1900;
    b.tm_mon = month_ - 1;
    b.tm_mday = day_;
    b.tm_min = b.tm_hour = b.tm_sec = 0;
    std::time_t x = std::mktime(&b);

    auto box_ptr = BoxWrapper::GetInstance();
    boxps::PSAgentBase* agent = box_ptr->GetAgent();
    VLOG(3) << "Begin call BeginFeedPass in BoxPS";
    box_ptr->BeginFeedPass(x / 86400, &agent);
    timer.Pause();

    feed_pass_span = timer.ElapsedSec();

    timer.Start();
    // add 0 key
    agent->AddKey(0ul, 0);
    dataset_->LoadIntoMemory();
    timer.Pause();
    read_ins_span = timer.ElapsedSec();

    timer.Start();
    // auc runner
    if (box_ptr->Mode() == 1) {
      box_ptr->AddReplaceFeasign(agent, box_ptr->GetFeedpassThreadNum());
    }
    box_ptr->EndFeedPass(agent);
#endif
    timer.Pause();

    VLOG(0) << "begin feedpass: " << feed_pass_span
            << "s, download + parse cost: " << read_ins_span
            << "s, end feedpass:" << timer.ElapsedSec() << "s";
  }

  void LoadIntoMemory() {
    platform::Timer timer;
    VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
    timer.Start();
    dataset_->LoadIntoMemory();
    timer.Pause();
    VLOG(0) << "download + parse cost: " << timer.ElapsedSec() << "s";

    timer.Start();
    FeedPass();
    timer.Pause();
    VLOG(0) << "FeedPass cost: " << timer.ElapsedSec() << " s";
    VLOG(3) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
  }
  void PreLoadIntoMemory() {
    dataset_->PreLoadIntoMemory();
    feed_data_thread_.reset(new std::thread([&]() {
      dataset_->WaitPreLoadDone();
      FeedPass();
    }));
    VLOG(3) << "After PreLoadIntoMemory()";
  }
  void WaitFeedPassDone() { feed_data_thread_->join(); }
  void SlotsShuffle(const std::set<std::string>& slots_to_replace) {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(box_ptr->Mode(), 1,
                      platform::errors::PreconditionNotMet(
                          "Should call InitForAucRunner first."));
    box_ptr->FlipPhase();

    std::unordered_set<uint16_t> index_slots;
    dynamic_cast<MultiSlotDataset*>(dataset_)->PreprocessChannel(
        slots_to_replace, index_slots);
    const std::vector<Record>& pass_data =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetSlotsOriginalData();
    if (!get_random_replace_done_) {
      box_ptr->GetRandomReplace(pass_data);
      get_random_replace_done_ = true;
    }
    std::vector<Record> random_data;
    random_data.resize(pass_data.size());
    box_ptr->GetRandomData(pass_data, index_slots, &random_data);

    auto new_input_channel = paddle::framework::MakeChannel<Record>();
    new_input_channel->Open();
    new_input_channel->Write(std::move(random_data));
    new_input_channel->Close();
    dynamic_cast<MultiSlotDataset*>(dataset_)->SetInputChannel(
        new_input_channel);
#endif
  }
#ifdef PADDLE_WITH_BOX_PS
  // notify boxps to feed this pass feasigns from SSD to memory
  static void FeedPassThread(const std::deque<Record>& t, int begin_index,
                             int end_index, boxps::PSAgentBase* p_agent,
                             const std::unordered_set<int>& index_map,
                             int thread_id) {
    p_agent->AddKey(0ul, thread_id);
    for (auto iter = t.begin() + begin_index; iter != t.begin() + end_index;
         iter++) {
      const auto& ins = *iter;
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        /*
        if (index_map.find(feasign.slot()) != index_map.end()) {
          continue;
        }
        */
        p_agent->AddKey(feasign.sign().uint64_feasign_, thread_id);
      }
    }
  }
#endif
  void FeedPass() {
    VLOG(3) << "Begin FeedPass";
#ifdef PADDLE_WITH_BOX_PS
    struct std::tm b;
    b.tm_year = year_ - 1900;
    b.tm_mon = month_ - 1;
    b.tm_mday = day_;
    b.tm_min = b.tm_sec = 0;
    b.tm_hour = FLAGS_fix_dayid ? 8 : 0;
    std::time_t x = std::mktime(&b);

    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    const std::deque<Record>& pass_data = input_channel_->GetData();

    // get feasigns that FeedPass doesn't need
    const std::unordered_set<std::string>& slot_name_omited_in_feedpass_ =
        box_ptr->GetOmitedSlot();
    std::unordered_set<int> slot_id_omited_in_feedpass_;
    const auto& all_readers = dataset_->GetReaders();
    PADDLE_ENFORCE_GT(all_readers.size(), 0,
                      platform::errors::PreconditionNotMet(
                          "Readers number must be greater than 0."));
    const auto& all_slots_name = all_readers[0]->GetAllSlotAlias();
    for (size_t i = 0; i < all_slots_name.size(); ++i) {
      if (slot_name_omited_in_feedpass_.find(all_slots_name[i]) !=
          slot_name_omited_in_feedpass_.end()) {
        slot_id_omited_in_feedpass_.insert(i);
      }
    }
    const size_t tnum = box_ptr->GetFeedpassThreadNum();
    boxps::PSAgentBase* p_agent = box_ptr->GetAgent();
    VLOG(3) << "Begin call BeginFeedPass in BoxPS";
    box_ptr->BeginFeedPass(x / 86400, &p_agent);

    std::vector<std::thread> threads;
    size_t len = pass_data.size();
    size_t len_per_thread = len / tnum;
    auto remain = len % tnum;
    size_t begin = 0;
    for (size_t i = 0; i < tnum; i++) {
      threads.push_back(
          std::thread(FeedPassThread, std::ref(pass_data), begin,
                      begin + len_per_thread + (i < remain ? 1 : 0), p_agent,
                      std::ref(slot_id_omited_in_feedpass_), i));
      begin += len_per_thread + (i < remain ? 1 : 0);
    }
    for (size_t i = 0; i < tnum; ++i) {
      threads[i].join();
    }

    if (box_ptr->Mode() == 1) {
      box_ptr->AddReplaceFeasign(p_agent, tnum);
    }
    VLOG(3) << "Begin call EndFeedPass in BoxPS";
    box_ptr->EndFeedPass(p_agent);
#endif
  }

 private:
  Dataset* dataset_;
  std::shared_ptr<std::thread> feed_data_thread_;
  int year_;
  int month_;
  int day_;
  bool get_random_replace_done_ = false;
};

}  // end namespace framework
}  // end namespace paddle

#include "paddle/fluid/framework/fleet/box_wrapper_impl.h"
