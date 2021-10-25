//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/gloo_context.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

void GLOOParallelContext::Init() {
  // PADDLE_THROW(platform::errors::OutOfRange(
  //  "Still not implement Init"));
  VLOG(4) << "Start GLOOParallelContext initialization";
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  gloo_wrapper->SetSize(strategy_.nranks_);
  gloo_wrapper->SetRank(strategy_.local_rank_);
  gloo_wrapper->SetPrefix("");
  gloo_wrapper->SetIface("lo");
  auto addr = paddle::string::Split(strategy_.trainer_endpoints_[0], ':');
  VLOG(4) << "Server is" << strategy_.trainer_endpoints_[0];
  std::string host = addr[0];
  int port = std::stoi(addr[1]);
  gloo_wrapper->SetHttpStore(host, port, "worker");
  gloo_wrapper->Init();
  device_ = std::unique_ptr<platform::CPUDeviceContext>(
      new platform::CPUDeviceContext(platform::CPUPlace()));
}

void GLOOParallelContext::InitWithRingID(int ring_id) {
  PADDLE_THROW(
      platform::errors::OutOfRange("Still not implement InitWithRingID"));
}

#define GLOO_CASE(type, T, gw)                                        \
  case type: {                                                        \
    VLOG(4) << "Use the gloo all reduce to sync. SRC:" << src_tensor; \
    std::vector<T> send_vector##T;                                    \
    framework::TensorToVector<T>(src_tensor, &send_vector##T);        \
    auto recv_vector##T = gw->AllReduce<T>(send_vector##T);           \
    framework::TensorFromVector<T>(recv_vector##T, dst_tensor);       \
    VLOG(4) << "DST:" << *dst_tensor;                                 \
    break;                                                            \
  }

void GLOOParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  // AllReduce(src, dst, strategy_, ring_id, use_calc_stream);
  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>());
  } else if (src.IsType<framework::SelectedRows>()) {
    if (&src != dst) {
      if (!dst->IsType<framework::SelectedRows>()) {
        dst->Clear();
      }
      AllReduce(src.Get<framework::SelectedRows>(),
                dst->GetMutable<framework::SelectedRows>());
    } else {
      // SelectedRows cannot be allreduce in-place
      framework::Variable tmp_dst;
      AllReduce(src.Get<framework::SelectedRows>(),
                tmp_dst.GetMutable<framework::SelectedRows>());
      *dst = std::move(tmp_dst);
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported variable type %s for imperative allreduce, only "
        "LoDTensor and SelectedRows are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

void GLOOParallelContext::AllReduce(const framework::Tensor &src_tensor,
                                    framework::Tensor *dst_tensor) {
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  dst_tensor->Resize(src_tensor.dims());
  switch (src_tensor.type()) {
    GLOO_CASE(framework::proto::VarType::FP32, float, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::FP64, double, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::INT32, int, gloo_wrapper);
    GLOO_CASE(framework::proto::VarType::INT64, int64_t, gloo_wrapper);
    default: {
      PADDLE_THROW(
          platform::errors::InvalidArgument("Invalid datatype for allreduce"));
    }
  }
  gloo_wrapper->Barrier();
}

#define GLOO_ALL_GATHER_CASE(type, T, gw)                         \
  case type: {                                                    \
    const auto *src_tensor_ptr = src_tensor.data<T>();            \
    gw->AllGatherVector<T>(const_cast<T *>(src_tensor_ptr),       \
                           reinterpret_cast<T *>(dst_tensor_ptr), \
                           value_sendcount);                      \
    break;                                                        \
  }

void GLOOParallelContext::AllReduce(const framework::SelectedRows &src,
                                    framework::SelectedRows *dst) {
  // auto ;
  // int local_rank = strategy_.local_rank_;
  int nranks = strategy_.nranks_;
  VLOG(3) << "SelectedRows AllReduce start";
  const auto &src_tensor = src.value();
  const auto &place = src_tensor.place();
  auto dtype = src_tensor.type();
  // 1. Gather rows number from all workers. Here use ncclAllGather to do this,
  // but we can use other ways to implement is in the future
  const auto &src_rows = src.rows();
  auto gloo_wrapper = framework::GlooWrapper::GetInstance();
  size_t local_row_num = src_rows.size();
  std::vector<size_t> rows_num_vector =
      gloo_wrapper->AllGather<size_t>(local_row_num);
  const auto *cpu_rows_num_ptr = rows_num_vector.data();
  auto rows_num = std::accumulate(cpu_rows_num_ptr, cpu_rows_num_ptr + nranks,
                                  static_cast<int64_t>(0));
  dst->set_height(src.height());
  VLOG(3) << "Gather rows: " << string::join_strings(rows_num_vector, ',')
          << ", total rows number: " << rows_num
          << ", height: " << src.height();
  auto *dst_rows = dst->mutable_rows();
  dst_rows->resize(rows_num);
  auto *dst_rows_ptr = dst_rows->MutableData(place);
  const int64_t *src_rows_ptr = src_rows.Data(place);

  // VLOG(3) << "Selected Rows of src:" << string::join_strings(dst_rows, ',')

  auto *dst_tensor = dst->mutable_value();
  auto dims = src_tensor.dims();
  dims[0] = rows_num;
  auto feature_size = framework::product(dims) / dims[0];
  dst_tensor->Resize(dims);
  if (std::all_of(cpu_rows_num_ptr, cpu_rows_num_ptr + nranks,
                  [&](size_t row) { return row == cpu_rows_num_ptr[0]; })) {
    // During sparse communication, the number of each card is same.
    // Because gloo wrapper utility class currently don't support
    // broadcast, so we only deal the-same case.
    VLOG(3) << "Use the gloo all reduce to sync. SRC:" << src_tensor;
    // framework::SerializeToStream(VLOG(4), src);
    VLOG(3) << "allgather replaces broadcast to speed up in sparse allreduce";
    auto value_sendcount = cpu_rows_num_ptr[0] * feature_size;
    auto *dst_tensor_ptr = dst_tensor->mutable_data(place, dtype);

    gloo_wrapper->AllGatherVector<int64_t>(const_cast<int64_t *>(src_rows_ptr),
                                           static_cast<int64_t *>(dst_rows_ptr),
                                           rows_num_vector[0]);

    switch (dtype) {
      GLOO_ALL_GATHER_CASE(framework::proto::VarType::FP32, float,
                           gloo_wrapper);
      GLOO_ALL_GATHER_CASE(framework::proto::VarType::FP64, double,
                           gloo_wrapper);
      GLOO_ALL_GATHER_CASE(framework::proto::VarType::INT32, int, gloo_wrapper);
      GLOO_ALL_GATHER_CASE(framework::proto::VarType::INT64, int64_t,
                           gloo_wrapper);
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid datatype for allreduce"));
      }
    }
    VLOG(3) << "Selected Row DST:" << *dst_tensor;
    VLOG(3) << "Selected Rows of DST:"
            << string::join_strings(std::vector<int64_t>(*dst_rows), ',');
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The number of each card is not the same, gloo only support the-same"
        "batch division"));
  }
}

paddle::platform::DeviceContext *GLOOParallelContext::GetDeviceContext(
    int ring_id) {
  // return the CPUDeviceContext
  return device_.get();
}

void GLOOParallelContext::WaitCompute(int ring_id) {
  // do nothing because cpu don't need sync
  return;
}

void GLOOParallelContext::WaitComm(int ring_id) {
  // do nothing because cpu don't need sync
  return;
}

void GLOOParallelContext::SynchronizeCompute() {
  // do nothing because cpu don't need sync
  return;
}

}  //  namespace imperative
}  //  namespace paddle
