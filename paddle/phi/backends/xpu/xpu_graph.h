// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.


#ifdef PADDLE_WITH_XPU


#include <thread>
#include <unordered_set>
#include <array>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include "paddle/common/errors.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/utils/optional.h"


#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/device_code.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"




namespace phi {
namespace backends {
namespace xpu {


class XPUGraphContextManager {
 public:
  using DeviceContextMap =
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>;


  static XPUGraphContextManager &Instance() {
    static XPUGraphContextManager instance;
    return instance;
  }


  DeviceContext *Get(int64_t pool_id, const Place &place, int stream_priority) {
    std::lock_guard<std::mutex> lk(ctx_mtx_);
    DeviceContextMap &ctxs = xpu_graph_ctx_pool_[pool_id];
    if (ctxs.find(place) == ctxs.end()) {
      phi::memory_utils::EmplaceDeviceContexts(
          &ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true,
          stream_priority);
    }
    return ctxs[place].get().get();
  }


  void RecordCapturingDeviceContext(DeviceContext *dev_ctx) {
    capturing_ctxs_.insert(dev_ctx);
  }


  std::set<DeviceContext *> GetAllCapturingDeviceContexts() const {
    return capturing_ctxs_;
  }


  void ClearDeviceContextsRecords() { capturing_ctxs_.clear(); }


 private:
  XPUGraphContextManager() = default;
  DISABLE_COPY_AND_ASSIGN(XPUGraphContextManager);


  std::mutex ctx_mtx_;
  std::unordered_map<int64_t, DeviceContextMap> xpu_graph_ctx_pool_;
  std::set<DeviceContext *> capturing_ctxs_;
};


class XPUKernelParams {
 public:
  explicit XPUKernelParams(void **params) : kernelParams(params) {}


  template <typename T>
  T &As(size_t idx) const {
    return *reinterpret_cast<T *>(kernelParams[idx]);
  }


  void **getParams() const {
    return kernelParams;
  }


 private:
  void **kernelParams;
};


class XPUGraphNodeLauncher {
 public:
  using parameterSetter_t = std::function<void(XPUKernelParams&)>;
 private:
  DISABLE_COPY_AND_ASSIGN(XPUGraphNodeLauncher);
};


using XPUGraphExecuterSetter_t = std::function<void(cudaGraphExec_t)>;  // 改用XPU图执行类型


static void ThrowErrorIfNotSupportXPUGraph() {
  PADDLE_THROW(common::errors::Unimplemented(
      "XPU Graph is not supported in current version"));
}


enum xpuStreamCaptureMode {
  xpuStreamCaptureModeGlobal = 0,
  xpuStreamCaptureModeThreadLocal = 1,
  xpuStreamCaptureModeRelaxed = 2
};


using XPUGraphID = unsigned long long;  // 重命名为XPU图ID类型


class XPUGraph {
  DISABLE_COPY_AND_ASSIGN(XPUGraph);
  XPUGraph();  // 构造函数保持私有
 public:
  using XPUPostResetCallback =
      std::function<void(paddle::optional<const XPUGraph&>)>;
  using XPUPreCaptureCallback = std::function<void()>;
  using XPUPostCaptureCallback = std::function<void()>;
  using SetSeedFunc = std::function<bool(XPUKernelParams *, bool)>;  // 适配XPUKernelParams


  static constexpr int64_t kDefaultPoolID = 0;
  static constexpr int64_t kInvalidPoolID = -1;


  static int64_t SetMemoryPoolID(int64_t pool_id);
  static void BeginCapture(phi::XPUPlace place,
                           XPUStream stream,  // 改用XPUStream
                           xpuStreamCaptureMode mode);  // 改用XPU流捕获模式
  static std::unique_ptr<XPUGraph> EndCapture();
  static void BeginSegmentCapture();
  static int64_t CapturingPoolID() { return capturing_graph_->pool_id_; }  // 关键：返回当前捕获的内存池ID
  static void EndSegmentCapture();
  static void AddJoiningStreamDuringCapturing(XPUStream stream);  // 改用XPUStream
  static void AddPostResetCallbackDuringCapturing(
      XPUPostResetCallback callback);
  static void AddPostCaptureCallbackDuringCapturing(
      XPUPostCaptureCallback callback);
  static bool IsCapturing();
  static XPUGraphID CapturingID();
  static phi::XPUPlace CapturingPlace();
  static bool IsValidCapturing();
  static bool IsThreadLocalCapturing();
  static bool IsThisThreadCapturing();  // 关键：判断当前线程是否在捕获中
  static void RecordRandomKernelInfo(SetSeedFunc set_seed_func);
  static int64_t UniqueMemoryPoolID();


  ~XPUGraph();
  XPUGraphID ID() const;
  int64_t PoolID() const;
  void Replay();
  void Reset();
  void AddPostResetCallback(XPUPostResetCallback callback);
  void AddPreCaptureCallback(XPUPreCaptureCallback callback);
  void AddPostCaptureCallback(XPUPostCaptureCallback callback);
  void AddJoiningStream(XPUStream stream);  // 改用XPUStream
  void PrintToDotFiles(const std::string &dirname, unsigned int flags);
  bool IsReplayed() const;


 private:
  static XPUGraphID UniqueID();


  std::vector<cudaGraph_t> graphs_;  // 改用XPU图类型
  std::vector<cudaGraphExec_t> exec_graphs_;  // 改用XPU图执行类型
  xpuStreamCaptureMode capture_mode_;  // 改用XPU流捕获模式


  XPUStream stream_{nullptr};  // 改用XPUStream
  phi::XPUPlace place_;
  XPUGraphID id_;
  int64_t pool_id_{kInvalidPoolID};
  bool is_reset_{false};
  bool is_replayed_{false};
  std::mutex mtx_;


  std::vector<SetSeedFunc> set_seed_funcs_;


  std::unordered_set<XPUStream> streams_to_join_;  // 改用XPUStream


  std::vector<std::function<void(paddle::optional<const XPUGraph &>)>>
      xpu_graph_post_reset_callbacks_;


  static std::vector<std::function<void()>> xpu_graph_pre_capture_callbacks_;
  std::vector<std::function<void()>> xpu_graph_post_capture_callbacks_;
  std::vector<std::vector<XPUGraphExecuterSetter_t>>
      xpu_graph_pre_replay_callbacks_;


  std::mutex func_mtx_;
  bool is_first_run_{true};


  static paddle::optional<std::thread::id> capturing_thread_id_;
  static std::unique_ptr<XPUGraph> capturing_graph_;  // 静态实例指向当前捕获的图
};


} // namespace xpu
} // namespace backends
} // namespace phi


#endif
