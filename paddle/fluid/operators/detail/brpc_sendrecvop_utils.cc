/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_CUDA
#include <nccl.h>
#endif
#include <sys/time.h>
#include <thread>  // NOLINT

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/detail/brpc_sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/brpc_variable_response.h"
#include "paddle/fluid/operators/detail/send_recv.pb.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace detail {

class IOBufWriter {
 public:
  static void Append(butil::IOBuf* iobuf, int k, const char* v, int64_t vlen) {
    iobuf->append(reinterpret_cast<char*>(&k), 4);
    iobuf->append(reinterpret_cast<char*>(&vlen), 8);
    // TODO(gongwb): use append_zero to avoid copy
    iobuf->append(v, vlen);
  }
};

void SerializeToIOBuf(const std::string& name, framework::Variable* var,
                      const platform::DeviceContext& ctx, VarMsg* request,
                      butil::IOBuf* iobuf, const std::string& out_varname) {
  void* payload = nullptr;
  size_t payload_size = 0;

  request->set_varname(name);
  // Note: normally the profiler is enabled in 1 trainer, hence only
  // 1 trainer returns true for ShouldSendProfileState(). It tells PS
  // servers the trainer's profiling state so that PS can follow the
  // trainer.
  if (platform::ShouldSendProfileState()) {
    if (platform::IsProfileEnabled()) {
      request->set_profile(platform::kEnableProfiler);
    } else {
      request->set_profile(platform::kDisableProfiler);
    }
  }
  if (!out_varname.empty()) {
    request->set_out_varname(out_varname);
  }
  if (var->IsType<framework::LoDTensor>()) {
    request->set_type(::sendrecv::LOD_TENSOR);
    GetTensorPayload(var, ctx, request, &payload, &payload_size);
  } else if (var->IsType<framework::SelectedRows>()) {
    request->set_type(::sendrecv::SELECTED_ROWS);
    GetSelectedRowsPayload(var, ctx, request, &payload, &payload_size);
#ifdef PADDLE_WITH_CUDA
  } else if (var->IsType<ncclUniqueId>()) {
    request->set_type(::sendrecv::NCCL_ID);
    const ncclUniqueId& uid = var->Get<ncclUniqueId>();
    // TODO(gongwb): use append_zero to avoid data copy.
    IOBufWriter::Append(iobuf,
                        sendrecv::VariableMessage::kSerializedFieldNumber,
                        uid.internal, NCCL_UNIQUE_ID_BYTES);
    return;
#endif
  } else {
    PADDLE_THROW("Serialize does not support type: %s",
                 typeid(var->Type()).name());
  }

  // TODO(gongwb): use append_zero to avoid data copy.
  IOBufWriter::Append(iobuf,
                      ::sendrecv::VariableMessage::kSerializedFieldNumber,
                      static_cast<char*>(payload), payload_size);

  if (var->IsType<framework::SelectedRows>()) {
    auto* slr = var->GetMutable<framework::SelectedRows>();
    size_t rows_memory_size =
        slr->rows().size() * framework::SizeOfType(typeid(int64_t));

    IOBufWriter::Append(iobuf, ::sendrecv::VariableMessage::kRowsFieldNumber,
                        reinterpret_cast<const char*>(slr->rows().data()),
                        static_cast<int64_t>(rows_memory_size));
  }
}

void DeserializeFromIOBuf(const ::sendrecv::VariableMessage& meta,
                          const butil::IOBuf& iobuf,
                          const platform::DeviceContext& ctx,
                          const framework::Scope* scope,
                          framework::Variable** var) {
  operators::detail::BRPCVariableResponse resp(scope, &ctx);
  PADDLE_ENFORCE(resp.Parse(iobuf, meta) == 0, "parse iobuf to tensor error!");
  *var = resp.GetVar();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
