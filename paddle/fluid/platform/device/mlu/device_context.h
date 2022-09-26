/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_MLU
#include <mutex>

#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/device/mlu/mlu_stream.h"
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_CNCL
#include <cncl.h>
#endif

namespace Eigen {
struct DefaultDevice;
struct GpuDevice;
}  // namespace Eigen

namespace paddle {
namespace platform {

class MLUContext {
 public:
  MLUContext() = default;
  explicit MLUContext(const MLUPlace& place, const int priority = 0);

  ~MLUContext();

  const MLUPlace& Place() const { return place_; }

  const std::unique_ptr<Eigen::DefaultDevice>& EigenDevice() const {
    return eigen_device_;
  }

  const std::unique_ptr<stream::MLUStream>& Stream() const { return stream_; }

  stream::MLUStream* SetStream(stream::MLUStream* new_stream_ptr) {
    auto* old_stream_ptr = stream_.release();
    stream_.reset(new_stream_ptr);
    return old_stream_ptr;
  }

  const mluStream& RawStream() { return stream_->raw_stream(); }

  const mluCnnlHandle& CnnlHandle() const { return cnnl_handle_; }

  const mluOpHandle& MluOpHandle() const { return mluOp_handle_; }

 private:
  void InitCNNLContext() {
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlCreate(&cnnl_handle_));
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlSetQueue(cnnl_handle_, RawStream()));
  }

  void InitMLUOPContext() {
    PADDLE_ENFORCE_MLU_SUCCESS(mluOpCreate(&mluOp_handle_));
    PADDLE_ENFORCE_MLU_SUCCESS(mluOpSetQueue(mluOp_handle_, RawStream()));
  }

  void DestoryCNNLContext() {
    if (cnnl_handle_) {
      PADDLE_ENFORCE_MLU_SUCCESS(cnnlDestroy(cnnl_handle_));
    }
    cnnl_handle_ = nullptr;
  }

  void DestoryMLUOPContext() {
    if (mluOp_handle_) {
      PADDLE_ENFORCE_MLU_SUCCESS(mluOpDestroy(mluOp_handle_));
    }
    mluOp_handle_ = nullptr;
  }

  MLUPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
  std::unique_ptr<stream::MLUStream> stream_;
  mluCnnlHandle cnnl_handle_;
  mluOpHandle mluOp_handle_;

  DISABLE_COPY_AND_ASSIGN(MLUContext);
};

class MLUDeviceContext : public DeviceContext {
 public:
  explicit MLUDeviceContext(MLUPlace place);
  virtual ~MLUDeviceContext();
  Eigen::DefaultDevice* eigen_device() const { return nullptr; }
  const Place& GetPlace() const override;

  int GetComputeCapability() const;

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Return cnnl handle in the device context. */
  mluCnnlHandle cnnl_handle() const;

  /*! \brief  Return mluOp handle in the device context. */
  mluOpHandle mluOp_handle() const;

  /*! \brief  Return mlu stream in the device context. */
  mluStream stream() const;

#ifdef PADDLE_WITH_CNCL
  /*! \brief  Return cncl communicators. */
  cnclComm_t cncl_comm() const { return cncl_comm_; }

  /*! \brief  Set cncl communicators. */
  void set_cncl_comm(cnclComm_t comm) { cncl_comm_ = comm; }
#endif

  template <typename Callback>
  void RecordEvent(mluEventHandle ev, Callback callback) const {
    return context()->Stream()->RecordEvent(ev, callback);
  }

  template <typename Callback>
  void AddStreamCallback(Callback&& callback) const {
    return context()->Stream()->AddCallback(callback);
  }

  void WaitStreamCallback() const {
    return context()->Stream()->WaitCallback();
  }

  void ResetDefaultContext(const int priority) {
    default_ctx_.reset(new MLUContext(place_, priority));
  }

  void ResetThreadContext(const int priority) {
    std::lock_guard<std::mutex> guard(ctx_mtx_);
    thread_ctx_[this].reset(new MLUContext(place_, priority));
  }

  std::shared_ptr<MLUContext> context() const {
    if (!thread_ctx_.count(this)) {
      return default_ctx_;
    }
    return thread_ctx_.at(this);
  }

 private:
  int compute_capability_;
  int driver_version_;
  int runtime_version_;
  int cnnl_version_;
  int mluOp_version_;
  MLUPlace place_;
  std::shared_ptr<MLUContext> default_ctx_;

  // The thread_local static variable will be released before the
  // global static variable, so avoid using it in dtor.
  static thread_local std::unordered_map<const MLUDeviceContext*,
                                         std::shared_ptr<MLUContext>>
      thread_ctx_;
  static thread_local std::mutex ctx_mtx_;

#ifdef PADDLE_WITH_CNCL
  cnclComm_t cncl_comm_{nullptr};
#endif

  DISABLE_COPY_AND_ASSIGN(MLUDeviceContext);
};

template <>
struct DefaultDeviceContextType<platform::MLUPlace> {
  using TYPE = MLUDeviceContext;
};

#endif

}  // namespace platform
}  // namespace paddle
