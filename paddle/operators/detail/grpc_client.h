#pragma once

#include <time.h>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
namespace detail {

struct Msg {
  std::string endpoint;
  std::string name;
  std::function<bool(std::string, std::string)> call_back;
  std::string error;
  // std::string time;
  time_t start;
}

class GRPCClient {
 public:
  GRPCClient() {
    channel_.reset(grpc::CreateChannel(ep, grpc::InsecureChannelCredentials()));
  }

  bool send(const framework::Scope& scope, std::vector<Msg>& in) {
    CompletionQueue cq;
    // Create a ClientContext, Status, Reply, and rpc for each backend.
    std::vector<std::unique_ptr<Service::Stub>> stubs;
    std::vector<std::unique_ptr<ClientContext>> contexts;
    std::vector<std::unique_ptr<Status>> statuses;
    std::vector<std::unique_ptr<Reply>> replies;
    std::vector<std::unique_ptr<ClientAsyncResponseReader<Reply>>> rpcs;

    int i = 0;
    for (int i = 0; i < in.size(); i++)
      stubs[i] = SendRecvService::NewStub(SendRecvService::NewStub(channel_));

    const auto start_time = chrono::system_clock::now();
    const chrono::system_clock::time_point deadline =
        start_time + chrono::milliseconds(5000);

    ClientContext* context = new ClientContext();
    context->set_deadline(deadline);
    contexts.emplace_back(context);

    statuses.emplace_back(new Status());

    Reply* reply = new Reply();
    replies->emplace_back(reply);

    rpcs.emplace_back(stubs_[ep.first]->AsyncFooCall(context, request, &cq));
    rpcs[i]->Finish(reply, statuses[i].get(), (void*)i);
    i++;
  }

  typedef std::map<std::string, std::string>::iterator end_type;

  int num_rpcs_finished = 0;
  int num_rpcs_finished_ok = 0;
  while (num_rpcs_finished < in.size()) {
    void* which;
    bool ok = false;
    // Block until the next result is available in the completion queue "cq".
    cq.Next(&which, &ok);
    num_rpcs_finished++;

    const int idx = int(which);
    const Status& status = *(statuses[idx].get());
    LOG(info) << "rpc #" << idx << " done after " << elapsed_ms(start_time)
              << "ms";

    if (status.ok()) {
      LOG(info) << "rpc ok";
      num_rpcs_finished_ok++;
    } else {
      if (status.error_code() == StatusCode::DEADLINE_EXCEEDED) {
        LOG(error) << "rpc timed out";
      } else {
        LOG(error) << "rpc failed because:" << status.error_code();
      }
    }
  }

  return true;
}


  bool GetVariable(const framework::Scope &scope,
          const std::string &outname);
{}

void Wait();

private:
std::shared_ptr<Channel> channel_
};
}
}
}
;
