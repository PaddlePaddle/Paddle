#include "paddle/fluid/framework/fleet/gloo_wrapper.h"

namespace paddle {
namespace framework {

template void GlooWrapper::AllReduce<int64_t>(
    const std::vector<int64_t>& sendbuf, std::vector<int64_t>& recvbuf);
template void GlooWrapper::AllReduce<double>(
    const std::vector<double>& sendbuf, std::vector<double>& recvbuf);
template std::vector<int64_t> GlooWrapper::AllGather<int64_t>(
    const int64_t& input);
template std::vector<double> GlooWrapper::AllGather<double>(
    const double& input);

}
}
