#pragma once

#include <time.h>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "paddle/framework/data_type.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/operators/detail/simple_block_queue.h"

#include "paddle/operators/detail/send_recv.grpc.pb.h"
#include "paddle/operators/detail/send_recv.pb.h"

#include <grpc++/grpc++.h>

using grpc::Channel;
using grpc::Server;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerBuilder;

using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using sendrecv::SendRecvService;
using sendrecv::VariableMessage;
using sendrecv::VoidMessage;

namespace paddle {
namespace operators {
namespace detail {

struct SendVar {
  std::string endpoint;
  std::string name;
  // std::function<bool(std::string, std::string)> call_back;
};

struct SendStatus {
  std::string error;
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  SendVar var;
};

class GRPCClient {
 public:
  GRPCClient() {}

  void AddEndPoint(std::string ep) {
    if (channels_.find(ep) != channels_.end()) {
      return;
    }

    channels_[ep] = std::shared_ptr<Channel>(
        grpc::CreateChannel(ep, grpc::InsecureChannelCredentials()));
  }

  void GetVariableMessage(const framework::Scope& scope, const std::string name,
                          VariableMessage* msg) {
    // FIXME(typhoonzero): pass device context to here.
    auto ctx = platform::CPUDeviceContext();
    auto* var = scope.FindVar(name);
    PADDLE_ENFORCE(var);
    // TODO(typhoonzero): support SelectedRows
    PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                   "Only support LoDTensor, %s has wrong type", name);

    const framework::LoDTensor& tensor = var->Get<framework::LoDTensor>();
    std::ostringstream oss;
    framework::SerializeToStream(oss, tensor, ctx);
    msg->set_varname(name);
    msg->set_serialized(oss.str());
  }

  // Send all variables of in.
  // if all are completed, return true, else false;
  bool Send(const framework::Scope& scope, std::vector<SendVar>& in,
            std::vector<SendStatus>& ret) {
    grpc::CompletionQueue cq;
    // Create a ClientContext, Status, Reply, and rpc for each backend.
    std::vector<std::unique_ptr<SendRecvService::Stub>> stubs;
    std::vector<std::unique_ptr<ClientContext>> contexts;
    std::vector<std::unique_ptr<Status>> statuses;
    std::vector<std::unique_ptr<VariableMessage>> replies;
    std::vector<
        std::unique_ptr<grpc::ClientAsyncResponseReader<VariableMessage>>>
        rpcs;

    typedef std::chrono::system_clock::time_point time_point;

    for (int i = 0; i < (int)in.size(); i++) {
      PADDLE_ENFORCE(channels_.find(in[i].endpoint) != channels_.end());
      auto ch = channels_[in[i].endpoint];
      stubs.emplace_back(SendRecvService::NewStub(ch));

      // record send status
      SendStatus s;
      s.start = std::chrono::system_clock::now();
      s.end = s.start;
      ret.emplace_back(s);
      const time_point deadline = s.start + std::chrono::milliseconds(5000);

      // context
      ClientContext* context = new ClientContext();
      context->set_deadline(deadline);
      contexts.emplace_back(context);

      // status
      statuses.emplace_back(new Status());

      // reply
      replies.emplace_back(new VariableMessage);

      // request
      VariableMessage request;
      GetVariableMessage(scope, in[i].name, &request);

      // rpcs
      rpcs.emplace_back(stubs[i]->AsyncSendVariable(context, request, &cq));
      rpcs[i]->Finish(&request, statuses[i].get(), (void*)i);
    }

    int finished = 0;
    int finished_ok = 0;
    bool all_completed = true;
    while (finished < int(in.size())) {
      void* which = NULL;
      bool ok = false;

      // Block until the next result is available in the completion queue "cq".
      cq.Next(&which, &ok);
      finished++;

      const int idx = int(which);
      ret[idx].end = std::chrono::system_clock::now();
      const Status& status = *(statuses[idx].get());

      if (status.ok()) {
        finished_ok++;
      } else {
        all_completed = false;
        if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
          ret[idx].error = "rpc timed out";
        } else {
          std::ostringstream stringStream;
          stringStream << "rpc failed because:" << status.error_code();
          ret[idx].error = stringStream.str();
        }
      }
    }

    return all_completed;
  }

 private:
  std::map<std::string, std::shared_ptr<Channel>> channels_;
};
}
}
};
