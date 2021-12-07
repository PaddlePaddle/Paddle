// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/service/graph_brpc_client.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "Eigen/Dense"
#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/table/table.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/string/string_helper.h"
namespace paddle {
namespace distributed {

void GraphPsService_Stub::service(
    ::google::protobuf::RpcController *controller,
    const ::paddle::distributed::PsRequestMessage *request,
    ::paddle::distributed::PsResponseMessage *response,
    ::google::protobuf::Closure *done) {
  if (graph_service != NULL && local_channel == channel()) {
    // VLOG(0)<<"use local";
    task_pool->enqueue([this, controller, request, response, done]() -> int {
      this->graph_service->service(controller, request, response, done);
      return 0;
    });
  } else {
    // VLOG(0)<<"use server";
    PsService_Stub::service(controller, request, response, done);
  }
}

int GraphBrpcClient::get_server_index_by_id(uint64_t id) {
  int shard_num = get_shard_num();
  int shard_per_server = shard_num % server_size == 0
                             ? shard_num / server_size
                             : shard_num / server_size + 1;
  return id % shard_num / shard_per_server;
}

std::future<int32_t> GraphBrpcClient::get_node_feat(
    const uint32_t &table_id, const std::vector<uint64_t> &node_ids,
    const std::vector<std::string> &feature_names,
    std::vector<std::vector<std::string>> &res) {
  std::vector<int> request2server;
  std::vector<int> server2request(server_size, -1);
  for (int query_idx = 0; query_idx < node_ids.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_ids[query_idx]);
    if (server2request[server_index] == -1) {
      server2request[server_index] = request2server.size();
      request2server.push_back(server_index);
    }
  }
  size_t request_call_num = request2server.size();
  std::vector<std::vector<uint64_t>> node_id_buckets(request_call_num);
  std::vector<std::vector<int>> query_idx_buckets(request_call_num);
  for (int query_idx = 0; query_idx < node_ids.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_ids[query_idx]);
    int request_idx = server2request[server_index];
    node_id_buckets[request_idx].push_back(node_ids[query_idx]);
    query_idx_buckets[request_idx].push_back(query_idx);
  }

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num,
      [&, node_id_buckets, query_idx_buckets, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (int request_idx = 0; request_idx < request_call_num;
             ++request_idx) {
          if (closure->check_response(request_idx, PS_GRAPH_GET_NODE_FEAT) !=
              0) {
            ++fail_num;
          } else {
            auto &res_io_buffer =
                closure->cntl(request_idx)->response_attachment();
            butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
            size_t bytes_size = io_buffer_itr.bytes_left();
            std::unique_ptr<char[]> buffer_wrapper(new char[bytes_size]);
            char *buffer = buffer_wrapper.get();
            io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);

            for (size_t feat_idx = 0; feat_idx < feature_names.size();
                 ++feat_idx) {
              for (size_t node_idx = 0;
                   node_idx < query_idx_buckets.at(request_idx).size();
                   ++node_idx) {
                int query_idx = query_idx_buckets.at(request_idx).at(node_idx);
                size_t feat_len = *(size_t *)(buffer);
                buffer += sizeof(size_t);
                auto feature = std::string(buffer, feat_len);
                res[feat_idx][query_idx] = feature;
                buffer += feat_len;
              }
            }
          }
          if (fail_num == request_call_num) {
            ret = -1;
          }
        }
        closure->set_promise_value(ret);
      });

  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (int request_idx = 0; request_idx < request_call_num; ++request_idx) {
    int server_index = request2server[request_idx];
    closure->request(request_idx)->set_cmd_id(PS_GRAPH_GET_NODE_FEAT);
    closure->request(request_idx)->set_table_id(table_id);
    closure->request(request_idx)->set_client_id(_client_id);
    size_t node_num = node_id_buckets[request_idx].size();

    closure->request(request_idx)
        ->add_params((char *)node_id_buckets[request_idx].data(),
                     sizeof(uint64_t) * node_num);
    std::string joint_feature_name =
        paddle::string::join_strings(feature_names, '\t');
    closure->request(request_idx)
        ->add_params(joint_feature_name.c_str(), joint_feature_name.size());

    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(request_idx)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(request_idx), closure->request(request_idx),
                     closure->response(request_idx), closure);
  }

  return fut;
}

std::future<int32_t> GraphBrpcClient::clear_nodes(uint32_t table_id) {
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      server_size, [&, server_size = this->server_size ](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (size_t request_idx = 0; request_idx < server_size; ++request_idx) {
          if (closure->check_response(request_idx, PS_GRAPH_CLEAR) != 0) {
            ++fail_num;
            break;
          }
        }
        ret = fail_num == 0 ? 0 : -1;
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < server_size; i++) {
    int server_index = i;
    closure->request(server_index)->set_cmd_id(PS_GRAPH_CLEAR);
    closure->request(server_index)->set_table_id(table_id);
    closure->request(server_index)->set_client_id(_client_id);

    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(server_index)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(server_index),
                     closure->request(server_index),
                     closure->response(server_index), closure);
  }
  return fut;
}
std::future<int32_t> GraphBrpcClient::add_graph_node(
    uint32_t table_id, std::vector<uint64_t> &node_id_list,
    std::vector<bool> &is_weighted_list) {
  std::vector<std::vector<uint64_t>> request_bucket;
  std::vector<std::vector<bool>> is_weighted_bucket;
  bool add_weight = is_weighted_list.size() > 0;
  std::vector<int> server_index_arr;
  std::vector<int> index_mapping(server_size, -1);
  for (size_t query_idx = 0; query_idx < node_id_list.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_id_list[query_idx]);
    if (index_mapping[server_index] == -1) {
      index_mapping[server_index] = request_bucket.size();
      server_index_arr.push_back(server_index);
      request_bucket.push_back(std::vector<uint64_t>());
      if (add_weight) is_weighted_bucket.push_back(std::vector<bool>());
    }
    request_bucket[index_mapping[server_index]].push_back(
        node_id_list[query_idx]);
    if (add_weight)
      is_weighted_bucket[index_mapping[server_index]].push_back(
          query_idx < is_weighted_list.size() ? is_weighted_list[query_idx]
                                              : false);
  }
  size_t request_call_num = request_bucket.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [&, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (size_t request_idx = 0; request_idx < request_call_num;
             ++request_idx) {
          if (closure->check_response(request_idx, PS_GRAPH_ADD_GRAPH_NODE) !=
              0) {
            ++fail_num;
          }
        }
        ret = fail_num == request_call_num ? -1 : 0;
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (size_t request_idx = 0; request_idx < request_call_num; ++request_idx) {
    int server_index = server_index_arr[request_idx];
    closure->request(request_idx)->set_cmd_id(PS_GRAPH_ADD_GRAPH_NODE);
    closure->request(request_idx)->set_table_id(table_id);
    closure->request(request_idx)->set_client_id(_client_id);
    size_t node_num = request_bucket[request_idx].size();
    closure->request(request_idx)
        ->add_params((char *)request_bucket[request_idx].data(),
                     sizeof(uint64_t) * node_num);
    if (add_weight) {
      bool weighted[is_weighted_bucket[request_idx].size() + 1];
      for (size_t j = 0; j < is_weighted_bucket[request_idx].size(); j++)
        weighted[j] = is_weighted_bucket[request_idx][j];
      closure->request(request_idx)
          ->add_params((char *)weighted,
                       sizeof(bool) * is_weighted_bucket[request_idx].size());
    }
    // PsService_Stub rpc_stub(get_cmd_channel(server_index));
    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(request_idx)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(request_idx), closure->request(request_idx),
                     closure->response(request_idx), closure);
  }
  return fut;
}
std::future<int32_t> GraphBrpcClient::remove_graph_node(
    uint32_t table_id, std::vector<uint64_t> &node_id_list) {
  std::vector<std::vector<uint64_t>> request_bucket;
  std::vector<int> server_index_arr;
  std::vector<int> index_mapping(server_size, -1);
  for (size_t query_idx = 0; query_idx < node_id_list.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_id_list[query_idx]);
    if (index_mapping[server_index] == -1) {
      index_mapping[server_index] = request_bucket.size();
      server_index_arr.push_back(server_index);
      request_bucket.push_back(std::vector<uint64_t>());
    }
    request_bucket[index_mapping[server_index]].push_back(
        node_id_list[query_idx]);
  }
  size_t request_call_num = request_bucket.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [&, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        int fail_num = 0;
        for (size_t request_idx = 0; request_idx < request_call_num;
             ++request_idx) {
          if (closure->check_response(request_idx,
                                      PS_GRAPH_REMOVE_GRAPH_NODE) != 0) {
            ++fail_num;
          }
        }
        ret = fail_num == request_call_num ? -1 : 0;
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (size_t request_idx = 0; request_idx < request_call_num; ++request_idx) {
    int server_index = server_index_arr[request_idx];
    closure->request(request_idx)->set_cmd_id(PS_GRAPH_REMOVE_GRAPH_NODE);
    closure->request(request_idx)->set_table_id(table_id);
    closure->request(request_idx)->set_client_id(_client_id);
    size_t node_num = request_bucket[request_idx].size();

    closure->request(request_idx)
        ->add_params((char *)request_bucket[request_idx].data(),
                     sizeof(uint64_t) * node_num);
    // PsService_Stub rpc_stub(get_cmd_channel(server_index));
    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(request_idx)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(request_idx), closure->request(request_idx),
                     closure->response(request_idx), closure);
  }
  return fut;
}
// char* &buffer,int &actual_size
std::future<int32_t> GraphBrpcClient::batch_sample_neighbors(
    uint32_t table_id, std::vector<uint64_t> node_ids, int sample_size,
    // std::vector<std::vector<std::pair<uint64_t, float>>> &res,
    std::vector<std::vector<uint64_t>> &res,
    std::vector<std::vector<float>> &res_weight, bool need_weight,
    int server_index) {
  if (server_index != -1) {
    res.resize(node_ids.size());
    if (need_weight) {
      res_weight.resize(node_ids.size());
    }
    DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [&](void *done) {
      int ret = 0;
      auto *closure = (DownpourBrpcClosure *)done;
      if (closure->check_response(0, PS_GRAPH_SAMPLE_NODES_FROM_ONE_SERVER) !=
          0) {
        ret = -1;
      } else {
        auto &res_io_buffer = closure->cntl(0)->response_attachment();
        butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
        size_t bytes_size = io_buffer_itr.bytes_left();
        std::unique_ptr<char[]> buffer_wrapper(new char[bytes_size]);
        char *buffer = buffer_wrapper.get();
        io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);

        size_t node_num = *(size_t *)buffer;
        int *actual_sizes = (int *)(buffer + sizeof(size_t));
        char *node_buffer = buffer + sizeof(size_t) + sizeof(int) * node_num;

        int offset = 0;
        for (size_t node_idx = 0; node_idx < node_num; ++node_idx) {
          int actual_size = actual_sizes[node_idx];
          int start = 0;
          while (start < actual_size) {
            res[node_idx].emplace_back(
                *(uint64_t *)(node_buffer + offset + start));
            start += GraphNode::id_size;
            if (need_weight) {
              res_weight[node_idx].emplace_back(
                  *(float *)(node_buffer + offset + start));
              start += GraphNode::weight_size;
            }
          }
          offset += actual_size;
        }
      }
      closure->set_promise_value(ret);
    });
    auto promise = std::make_shared<std::promise<int32_t>>();
    closure->add_promise(promise);
    std::future<int> fut = promise->get_future();
    ;
    closure->request(0)->set_cmd_id(PS_GRAPH_SAMPLE_NODES_FROM_ONE_SERVER);
    closure->request(0)->set_table_id(table_id);
    closure->request(0)->set_client_id(_client_id);
    closure->request(0)->add_params((char *)node_ids.data(),
                                    sizeof(uint64_t) * node_ids.size());
    closure->request(0)->add_params((char *)&sample_size, sizeof(int));
    closure->request(0)->add_params((char *)&need_weight, sizeof(bool));
    ;
    // PsService_Stub rpc_stub(get_cmd_channel(server_index));
    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(0), closure->request(0),
                     closure->response(0), closure);
    return fut;
  }
  std::vector<int> request2server;
  std::vector<int> server2request(server_size, -1);
  res.clear();
  res_weight.clear();
  for (int query_idx = 0; query_idx < node_ids.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_ids[query_idx]);
    if (server2request[server_index] == -1) {
      server2request[server_index] = request2server.size();
      request2server.push_back(server_index);
    }
    // res.push_back(std::vector<std::pair<uint64_t, float>>());
    res.push_back({});
    if (need_weight) {
      res_weight.push_back({});
    }
  }
  size_t request_call_num = request2server.size();
  std::vector<std::vector<uint64_t>> node_id_buckets(request_call_num);
  std::vector<std::vector<int>> query_idx_buckets(request_call_num);
  for (int query_idx = 0; query_idx < node_ids.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_ids[query_idx]);
    int request_idx = server2request[server_index];
    node_id_buckets[request_idx].push_back(node_ids[query_idx]);
    query_idx_buckets[request_idx].push_back(query_idx);
  }

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num,
      [&, node_id_buckets, query_idx_buckets, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (size_t request_idx = 0; request_idx < request_call_num;
             ++request_idx) {
          if (closure->check_response(request_idx, PS_GRAPH_SAMPLE_NEIGHBORS) !=
              0) {
            ++fail_num;
          } else {
            auto &res_io_buffer =
                closure->cntl(request_idx)->response_attachment();
            butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
            size_t bytes_size = io_buffer_itr.bytes_left();
            std::unique_ptr<char[]> buffer_wrapper(new char[bytes_size]);
            char *buffer = buffer_wrapper.get();
            io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);

            size_t node_num = *(size_t *)buffer;
            int *actual_sizes = (int *)(buffer + sizeof(size_t));
            char *node_buffer =
                buffer + sizeof(size_t) + sizeof(int) * node_num;

            int offset = 0;
            for (size_t node_idx = 0; node_idx < node_num; ++node_idx) {
              int query_idx = query_idx_buckets.at(request_idx).at(node_idx);
              int actual_size = actual_sizes[node_idx];
              int start = 0;
              while (start < actual_size) {
                res[query_idx].emplace_back(
                    *(uint64_t *)(node_buffer + offset + start));
                start += GraphNode::id_size;
                if (need_weight) {
                  res_weight[query_idx].emplace_back(
                      *(float *)(node_buffer + offset + start));
                  start += GraphNode::weight_size;
                }
              }
              offset += actual_size;
            }
          }
          if (fail_num == request_call_num) {
            ret = -1;
          }
        }
        closure->set_promise_value(ret);
      });

  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (int request_idx = 0; request_idx < request_call_num; ++request_idx) {
    int server_index = request2server[request_idx];
    closure->request(request_idx)->set_cmd_id(PS_GRAPH_SAMPLE_NEIGHBORS);
    closure->request(request_idx)->set_table_id(table_id);
    closure->request(request_idx)->set_client_id(_client_id);
    size_t node_num = node_id_buckets[request_idx].size();

    closure->request(request_idx)
        ->add_params((char *)node_id_buckets[request_idx].data(),
                     sizeof(uint64_t) * node_num);
    closure->request(request_idx)
        ->add_params((char *)&sample_size, sizeof(int));
    closure->request(request_idx)
        ->add_params((char *)&need_weight, sizeof(bool));
    // PsService_Stub rpc_stub(get_cmd_channel(server_index));
    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(request_idx)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(request_idx), closure->request(request_idx),
                     closure->response(request_idx), closure);
  }

  return fut;
}
std::future<int32_t> GraphBrpcClient::random_sample_nodes(
    uint32_t table_id, int server_index, int sample_size,
    std::vector<uint64_t> &ids) {
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [&](void *done) {
    int ret = 0;
    auto *closure = (DownpourBrpcClosure *)done;
    if (closure->check_response(0, PS_GRAPH_SAMPLE_NODES) != 0) {
      ret = -1;
    } else {
      auto &res_io_buffer = closure->cntl(0)->response_attachment();
      butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
      size_t bytes_size = io_buffer_itr.bytes_left();
      char *buffer = new char[bytes_size];
      auto size = io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);
      int index = 0;
      while (index < bytes_size) {
        ids.push_back(*(uint64_t *)(buffer + index));
        index += GraphNode::id_size;
      }
      delete[] buffer;
    }
    closure->set_promise_value(ret);
  });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  ;
  closure->request(0)->set_cmd_id(PS_GRAPH_SAMPLE_NODES);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  closure->request(0)->add_params((char *)&sample_size, sizeof(int));
  ;
  // PsService_Stub rpc_stub(get_cmd_channel(server_index));
  GraphPsService_Stub rpc_stub = getServiceStub(get_cmd_channel(server_index));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);
  return fut;
}

std::future<int32_t> GraphBrpcClient::load_graph_split_config(
    uint32_t table_id, std::string path) {
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      server_size, [&, server_size = this->server_size ](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (size_t request_idx = 0; request_idx < server_size; ++request_idx) {
          if (closure->check_response(request_idx,
                                      PS_GRAPH_LOAD_GRAPH_SPLIT_CONFIG) != 0) {
            ++fail_num;
            break;
          }
        }
        ret = fail_num == 0 ? 0 : -1;
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < server_size; i++) {
    int server_index = i;
    closure->request(server_index)
        ->set_cmd_id(PS_GRAPH_LOAD_GRAPH_SPLIT_CONFIG);
    closure->request(server_index)->set_table_id(table_id);
    closure->request(server_index)->set_client_id(_client_id);
    closure->request(server_index)->add_params(path);
    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(server_index)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(server_index),
                     closure->request(server_index),
                     closure->response(server_index), closure);
  }
  return fut;
}
std::future<int32_t> GraphBrpcClient::use_neighbors_sample_cache(
    uint32_t table_id, size_t total_size_limit, size_t ttl) {
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      server_size, [&, server_size = this->server_size ](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (size_t request_idx = 0; request_idx < server_size; ++request_idx) {
          if (closure->check_response(
                  request_idx, PS_GRAPH_USE_NEIGHBORS_SAMPLE_CACHE) != 0) {
            ++fail_num;
            break;
          }
        }
        ret = fail_num == 0 ? 0 : -1;
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  size_t size_limit = total_size_limit / server_size +
                      (total_size_limit % server_size != 0 ? 1 : 0);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < server_size; i++) {
    int server_index = i;
    closure->request(server_index)
        ->set_cmd_id(PS_GRAPH_USE_NEIGHBORS_SAMPLE_CACHE);
    closure->request(server_index)->set_table_id(table_id);
    closure->request(server_index)->set_client_id(_client_id);
    closure->request(server_index)
        ->add_params((char *)&size_limit, sizeof(size_t));
    closure->request(server_index)->add_params((char *)&ttl, sizeof(size_t));
    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(server_index)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(server_index),
                     closure->request(server_index),
                     closure->response(server_index), closure);
  }
  return fut;
}
std::future<int32_t> GraphBrpcClient::pull_graph_list(
    uint32_t table_id, int server_index, int start, int size, int step,
    std::vector<FeatureNode> &res) {
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [&](void *done) {
    int ret = 0;
    auto *closure = (DownpourBrpcClosure *)done;
    if (closure->check_response(0, PS_PULL_GRAPH_LIST) != 0) {
      ret = -1;
    } else {
      auto &res_io_buffer = closure->cntl(0)->response_attachment();
      butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
      size_t bytes_size = io_buffer_itr.bytes_left();
      char *buffer = new char[bytes_size];
      io_buffer_itr.copy_and_forward((void *)(buffer), bytes_size);
      int index = 0;
      while (index < bytes_size) {
        FeatureNode node;
        node.recover_from_buffer(buffer + index);
        index += node.get_size(false);
        res.push_back(node);
      }
      delete[] buffer;
    }
    closure->set_promise_value(ret);
  });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  closure->request(0)->set_cmd_id(PS_PULL_GRAPH_LIST);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  closure->request(0)->add_params((char *)&start, sizeof(int));
  closure->request(0)->add_params((char *)&size, sizeof(int));
  closure->request(0)->add_params((char *)&step, sizeof(int));
  // PsService_Stub rpc_stub(get_cmd_channel(server_index));
  GraphPsService_Stub rpc_stub = getServiceStub(get_cmd_channel(server_index));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);
  return fut;
}

std::future<int32_t> GraphBrpcClient::set_node_feat(
    const uint32_t &table_id, const std::vector<uint64_t> &node_ids,
    const std::vector<std::string> &feature_names,
    const std::vector<std::vector<std::string>> &features) {
  std::vector<int> request2server;
  std::vector<int> server2request(server_size, -1);
  for (int query_idx = 0; query_idx < node_ids.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_ids[query_idx]);
    if (server2request[server_index] == -1) {
      server2request[server_index] = request2server.size();
      request2server.push_back(server_index);
    }
  }
  size_t request_call_num = request2server.size();
  std::vector<std::vector<uint64_t>> node_id_buckets(request_call_num);
  std::vector<std::vector<int>> query_idx_buckets(request_call_num);
  std::vector<std::vector<std::vector<std::string>>> features_idx_buckets(
      request_call_num);
  for (int query_idx = 0; query_idx < node_ids.size(); ++query_idx) {
    int server_index = get_server_index_by_id(node_ids[query_idx]);
    int request_idx = server2request[server_index];
    node_id_buckets[request_idx].push_back(node_ids[query_idx]);
    query_idx_buckets[request_idx].push_back(query_idx);
    if (features_idx_buckets[request_idx].size() == 0) {
      features_idx_buckets[request_idx].resize(feature_names.size());
    }
    for (int feat_idx = 0; feat_idx < feature_names.size(); ++feat_idx) {
      features_idx_buckets[request_idx][feat_idx].push_back(
          features[feat_idx][query_idx]);
    }
  }

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num,
      [&, node_id_buckets, query_idx_buckets, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        size_t fail_num = 0;
        for (int request_idx = 0; request_idx < request_call_num;
             ++request_idx) {
          if (closure->check_response(request_idx, PS_GRAPH_SET_NODE_FEAT) !=
              0) {
            ++fail_num;
          }
          if (fail_num == request_call_num) {
            ret = -1;
          }
        }
        closure->set_promise_value(ret);
      });

  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (int request_idx = 0; request_idx < request_call_num; ++request_idx) {
    int server_index = request2server[request_idx];
    closure->request(request_idx)->set_cmd_id(PS_GRAPH_SET_NODE_FEAT);
    closure->request(request_idx)->set_table_id(table_id);
    closure->request(request_idx)->set_client_id(_client_id);
    size_t node_num = node_id_buckets[request_idx].size();

    closure->request(request_idx)
        ->add_params((char *)node_id_buckets[request_idx].data(),
                     sizeof(uint64_t) * node_num);
    std::string joint_feature_name =
        paddle::string::join_strings(feature_names, '\t');
    closure->request(request_idx)
        ->add_params(joint_feature_name.c_str(), joint_feature_name.size());

    // set features
    std::string set_feature = "";
    for (size_t feat_idx = 0; feat_idx < feature_names.size(); ++feat_idx) {
      for (size_t node_idx = 0; node_idx < node_num; ++node_idx) {
        size_t feat_len =
            features_idx_buckets[request_idx][feat_idx][node_idx].size();
        set_feature.append((char *)&feat_len, sizeof(size_t));
        set_feature.append(
            features_idx_buckets[request_idx][feat_idx][node_idx].data(),
            feat_len);
      }
    }
    closure->request(request_idx)
        ->add_params(set_feature.c_str(), set_feature.size());

    GraphPsService_Stub rpc_stub =
        getServiceStub(get_cmd_channel(server_index));
    closure->cntl(request_idx)->set_log_id(butil::gettimeofday_ms());
    rpc_stub.service(closure->cntl(request_idx), closure->request(request_idx),
                     closure->response(request_idx), closure);
  }

  return fut;
}

int32_t GraphBrpcClient::initialize() {
  // set_shard_num(_config.shard_num());
  BrpcPsClient::initialize();
  server_size = get_server_nums();
  graph_service = NULL;
  local_channel = NULL;
  return 0;
}
}
}
