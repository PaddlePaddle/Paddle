// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"
#include "glog/logging.h"
#include "paddle/common/flags.h"

#ifdef PADDLE_WITH_CUDA

#if CUDA_VERSION < 11000
cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                const void *symbolPtr) {
  return cudaSuccess;
}
#endif

COMMON_DECLARE_bool(use_cuda_malloc_async_allocator);
COMMON_DECLARE_bool(auto_free_cudagraph_allocations_on_launch);

namespace phi::backends::gpu {

std::unique_ptr<CUDAGraph> CUDAGraph::capturing_graph_{nullptr};
paddle::optional<std::thread::id> CUDAGraph::capturing_thread_id_{paddle::none};
std::vector<std::function<void()>> CUDAGraph::cudagraph_pre_capture_callbacks_;

static std::vector<cudaGraphNode_t> ToposortCUDAGraph(cudaGraph_t graph) {
  size_t num_nodes;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphGetNodes(graph, nodes.data(), &num_nodes));

  size_t num_edges;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphGetEdges(graph, nullptr, nullptr, &num_edges));
  std::vector<cudaGraphNode_t> from(num_edges), to(num_edges);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphGetEdges(graph, from.data(), to.data(), &num_edges));

  std::unordered_map<cudaGraphNode_t, std::unordered_set<cudaGraphNode_t>>
      in_edges, out_edges;
  for (auto node : nodes) {
    in_edges[node];
    out_edges[node];
  }

  for (size_t i = 0; i < num_edges; ++i) {
    in_edges[to[i]].insert(from[i]);
    out_edges[from[i]].insert(to[i]);
  }

  std::queue<cudaGraphNode_t> q;
  for (const auto &pair : in_edges) {
    if (pair.second.empty()) {
      q.push(pair.first);
    }
  }

  nodes.clear();
  while (!q.empty()) {
    auto cur = q.front();
    q.pop();
    nodes.push_back(cur);

    for (auto out_node : out_edges.at(cur)) {
      auto &in_nodes = in_edges.at(out_node);
      in_nodes.erase(cur);
      if (in_nodes.empty()) {
        q.push(out_node);
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      nodes.size(),
      num_nodes,
      common::errors::InvalidArgument("Toposort error, this may be a bug."));
  return nodes;
}

CUDAGraphID CUDAGraph::UniqueID() {
  static std::atomic<CUDAGraphID> id;
  return id.fetch_add(1);
}

int64_t CUDAGraph::UniqueMemoryPoolID() {
  static std::atomic<int64_t> id(CUDAGraph::kDefaultPoolID + 1);
  return id.fetch_add(1);
}

void CUDAGraph::Reset() {
  if (is_reset_) return;
#if CUDA_VERSION >= 10010
  for (auto graph : graphs_) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphDestroy(graph));
  }
  graphs_.clear();
  for (auto exec_graph : exec_graphs_) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphExecDestroy(exec_graph));
  }
  exec_graphs_.clear();
#endif
  // callback should be called in reverse order because the latter added
  // callback may rely on the former added callback.
  for (auto iter = cudagraph_post_reset_callbacks_.rbegin();
       iter != cudagraph_post_reset_callbacks_.rend();
       ++iter) {
    (*iter)(*this);
  }
  cudagraph_post_reset_callbacks_.clear();
  is_reset_ = true;
}

void CUDAGraph::Replay() {
  is_replayed_ = true;
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(is_reset_,
                    false,
                    common::errors::PermissionDenied(
                        "Cannot replay the CUDA Graph after reset is called."));
  size_t n = exec_graphs_.size();
  for (size_t i = 0; i < n; ++i) {
    if (!is_first_run_) {
      for (auto &hook : cudagraph_pre_replay_callbacks_[i]) {
        hook(exec_graphs_[i]);
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphLaunch(exec_graphs_[i], stream_));
  }
  is_first_run_ = false;
#endif
}

void CUDAGraph::BeginSegmentCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(IsCapturing(),
                    true,
                    common::errors::PermissionDenied(
                        "BeginSegmentCapture should be called when CUDA "
                        "Graph is capturing."));
  if (IsThreadLocalCapturing()) {
    PADDLE_ENFORCE_EQ(IsThisThreadCapturing(),
                      true,
                      common::errors::PermissionDenied(
                          "When capturing CUDA Graph in the thread local mode, "
                          "you cannot begin segmented capturing in the thread "
                          "which is not the one that starts the capturing."));
  }

  for (auto &hook : cudagraph_pre_capture_callbacks_) {
    hook();
  }

  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamBeginCapture(
      capturing_graph_->stream_, capturing_graph_->capture_mode_));
  PADDLE_ENFORCE_EQ(IsValidCapturing(),
                    true,
                    common::errors::PermissionDenied(
                        "CUDA Graph should not be invalidated."));
  VLOG(10) << "Begin to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
#endif
}

void CUDAGraph::BeginCapture(phi::GPUPlace place,
                             cudaStream_t stream,
                             cudaStreamCaptureMode mode) {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(IsCapturing(),
                    false,
                    common::errors::PermissionDenied(
                        "CUDA Graph can only captured one by one."));
  PADDLE_ENFORCE_NOT_NULL(
      stream,
      common::errors::PermissionDenied(
          "CUDA Graph cannot be captured in default CUDA stream 0."));
  capturing_graph_.reset(new CUDAGraph());
  capturing_graph_->place_ = place;
  capturing_graph_->stream_ = stream;
  capturing_graph_->capture_mode_ = mode;
  if (mode == cudaStreamCaptureModeThreadLocal) {
    capturing_thread_id_ = std::this_thread::get_id();
    VLOG(10) << "Capturing CUDA Graph in thread local mode, thread id: "
             << capturing_thread_id_;
  }
  BeginSegmentCapture();
#endif
}

inline void sync_streams(gpuStream_t to_record, gpuStream_t to_wait) {
  if (to_record == to_wait) return;
  cudaEvent_t event = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, to_record));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(to_wait, event));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventDestroy(event));
}

void CUDAGraph::EndSegmentCapture() {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 10010
  PADDLE_ENFORCE_EQ(
      IsCapturing(),
      true,
      common::errors::PermissionDenied("No CUDA Graph is capturing."));

  for (const auto &stream : capturing_graph_->streams_to_join_) {
    VLOG(10) << "Joining steam when the capture is going to end stream ="
             << stream;
    sync_streams(stream, capturing_graph_->stream_);
  }
  capturing_graph_->streams_to_join_.clear();

  cudaGraph_t graph;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamEndCapture(capturing_graph_->stream_, &graph));
  auto num_nodes = static_cast<size_t>(-1);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  if (num_nodes == 0) {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphDestroy(graph));
    VLOG(10) << "Skip empty CUDA Graph with ID " << capturing_graph_->id_
             << ", segment id " << capturing_graph_->graphs_.size()
             << ", memory pool id " << capturing_graph_->pool_id_;
    return;
  }

  for (auto &cudagraph_post_capture_callback :
       capturing_graph_->cudagraph_post_capture_callbacks_) {
    cudagraph_post_capture_callback();
  }
  capturing_graph_->cudagraph_post_capture_callbacks_.clear();

  capturing_graph_->cudagraph_pre_replay_callbacks_.emplace_back(
      CUDAGraphNodeLauncher::Instance().GetParameterSettersForExecGraph(graph));

  cudaGraphExec_t exec_graph;
  if (FLAGS_use_cuda_malloc_async_allocator &&
      FLAGS_auto_free_cudagraph_allocations_on_launch) {
#if CUDA_VERSION >= 11040
    VLOG(1) << "cudaGraphInstantiateFlagAutoFreeOnLaunch is enabled!";
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphInstantiateWithFlags(
        &exec_graph, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "The cudaGraphInstantiateFlagAutoFreeOnLaunch is only supported when "
        "CUDA version >= 11.4.0"));
#endif
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaGraphInstantiate(&exec_graph, graph, nullptr, nullptr, 0));
  }
  VLOG(10) << "End to capture CUDA Graph with ID " << capturing_graph_->id_
           << ", segment id " << capturing_graph_->graphs_.size()
           << ", memory pool id " << capturing_graph_->pool_id_;
  capturing_graph_->graphs_.emplace_back(graph);
  capturing_graph_->exec_graphs_.emplace_back(exec_graph);
#endif
}

std::unique_ptr<CUDAGraph> CUDAGraph::EndCapture() {
  EndSegmentCapture();
  capturing_thread_id_ = paddle::none;
  return std::move(capturing_graph_);
}

bool CUDAGraph::IsValidCapturing() {
#if CUDA_VERSION >= 10010
  if (!IsCapturing()) return false;
  cudaStreamCaptureStatus status;
  CUDAGraphID id;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaStreamGetCaptureInfo(capturing_graph_->stream_, &status, &id));
  return status == cudaStreamCaptureStatusActive;
#else
  return false;
#endif
}

static std::string ConcatPath(const std::string &dirname,
                              const std::string &filename) {
#ifdef _WIN32
  const std::array<char, 3> kFileSep = {"\\"};
#else
  const std::array<char, 2> kFileSep = {"/"};
#endif
  if (!dirname.empty() && dirname.back() == kFileSep[0]) {
    return dirname + filename;
  } else {
    return dirname + kFileSep.data() + filename;
  }
}

void CUDAGraph::PrintToDotFiles(const std::string &dirname,
                                unsigned int flags) {
  ThrowErrorIfNotSupportCUDAGraph();
#if CUDA_VERSION >= 11030
  for (size_t i = 0; i < graphs_.size(); ++i) {
    auto filename =
        ConcatPath(dirname, "segment_" + std::to_string(i) + ".dot");
    VLOG(10) << "Save the " << i << "-th segment of graph " << id_ << " to "
             << filename;
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaGraphDebugDotPrint(graphs_[i], filename.c_str(), flags));
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "The print_to_dot_files() method is only supported when CUDA version >= "
      "11.3."));
#endif
}

#if CUDA_VERSION >= 11000
void CUDAGraphNodeLauncher::KernelNodeLaunch(
    parameterSetter_t parameterSetter, gpuKernelCallback_t cudakernelCallback) {
  if (UNLIKELY(phi::backends::gpu::CUDAGraph::IsThisThreadCapturing())) {
    unsigned int id = GenerateIdentifier();
    auto cudaFunc = cudakernelCallback(id);

    parameterSetters[cudaFunc][id] = parameterSetter;
    VLOG(10) << "[KernelNodeLaunch] Launch kernel with cudaFunc = " << cudaFunc
             << " id = " << id;
  } else {
    cudakernelCallback(0);
  }
}

std::vector<cudaGraphExecuterSetter_t>
CUDAGraphNodeLauncher::GetParameterSettersForExecGraph(cudaGraph_t graph) {
  size_t num_nodes;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaGraphGetNodes(graph, nullptr, &num_nodes));
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaGraphGetNodes(graph, nodes.data(), &num_nodes));

  std::vector<std::function<void(cudaGraphExec_t)>> hooks;
  for (auto node : nodes) {
    CUgraphNode cuNode = node;
    CUgraphNodeType pType;
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cuGraphNodeGetType(cuNode, &pType));
    if (pType == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS cuParams;

      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cuGraphKernelNodeGetParams(cuNode, &cuParams));
      gpuKernelParams kernel_params(cuParams.kernelParams);
      auto kernel =
          parameterSetters.find(static_cast<cudaFunction_t>(cuParams.func));
      VLOG(10) << "[GetParameterSettersForExecGraph] cuParams.func = "
               << cuParams.func;
      // There exists a parameter setter
      if (kernel != parameterSetters.end()) {
        auto launchSequence = kernel->second;
        unsigned int id = kernel_params.As<int>(0);

        VLOG(10) << "[GetParameterSettersForExecGraph] Find launch kernel id = "
                 << id;
        auto parameterSetter = launchSequence.find(id);
        if (parameterSetter != launchSequence.end()) {
          auto setter = parameterSetter->second;
          hooks.emplace_back([setter, cuNode, cuParams](
                                 cudaGraphExec_t exec_graph) {
            gpuKernelParams kernel_params(cuParams.kernelParams);
            setter(kernel_params);
            PADDLE_ENFORCE_GPU_SUCCESS(dynload::cuGraphExecKernelNodeSetParams(
                static_cast<CUgraphExec>(exec_graph), cuNode, &cuParams));
          });
        } else {
          PADDLE_THROW(common::errors::InvalidArgument(
              "Error: does not find launch id"));
        }
      }
    }
  }

  return hooks;
}
#else
void CUDAGraphNodeLauncher::KernelNodeLaunch(
    cudaFunction_t cudaFunc,
    parameterSetter_t parameterSetter,
    gpuKernelCallback_t cudakernelCallback) {
  cudakernelCallback(0);
}

std::vector<cudaGraphExecuterSetter_t>
CUDAGraphNodeLauncher::GetParameterSettersForExecGraph(cudaGraph_t graph) {
  PADDLE_THROW(common::errors::Unimplemented(
      "CUDAGraphNodeLauncher is only supported when CUDA version >= 11.0"));
}
#endif

}  // namespace phi::backends::gpu

#endif
