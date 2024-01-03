#include "paddle/phi/core/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/device_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_mapper.h"

# define TO_PROTO_HELPER(object, proto_type) \
  proto_type proto; \
  object.to_proto(&proto); \
  return proto

namespace phi {
namespace distributed {


     auto_parallel::TensorDistAttrProto to_proto(const TensorDistAttr& dist_attr){
        TO_PROTO_HELPER(dist_attr, auto_parallel::TensorDistAttrProto);
    }

    auto_parallel::ProcessMeshProto to_proto(const ProcessMesh& process_mesh){
        TO_PROTO_HELPER(process_mesh, auto_parallel::ProcessMeshProto);
    }

    
    auto_parallel::DeviceCapabilityProto to_proto(const auto_parallel::DeviceCapability& device_capibilty){
        TO_PROTO_HELPER(device_capibilty, auto_parallel::DeviceCapabilityProto);
    }

    auto_parallel::DeviceProto to_proto(const auto_parallel::Device& device){
        TO_PROTO_HELPER(device, auto_parallel::DeviceProto);
    }

    auto_parallel::LinkCapabilityProto to_proto(const auto_parallel::LinkCapability& link_capibilty){
        TO_PROTO_HELPER(link_capibilty, auto_parallel::LinkCapabilityProto);
    }

    auto_parallel::LinkProto to_proto(const auto_parallel::Link& link){
        TO_PROTO_HELPER(link, auto_parallel::LinkProto);
    }

    auto_parallel::DeviceMeshProto to_proto(const auto_parallel::DeviceMesh& device_mesh){
        TO_PROTO_HELPER(device_mesh, auto_parallel::DeviceMeshProto);
    }

    auto_parallel::DistributedMapperProto to_proto(const auto_parallel::DistributedMapper& dist_mapper){
        TO_PROTO_HELPER(dist_mapper, auto_parallel::DistributedMapperProto);
    }
}
}