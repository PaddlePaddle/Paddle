#include "gflags/gflags.h"
#include "paddle/fluid/platform/mlu/enforce.h"
#include "paddle/fluid/platform/mlu/mlu_info.h"
#include "paddle/fluid/string/split.h"

PADDLE_DEFINE_EXPORTED_string(
    selected_mlus, "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (MLU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between MLU devices, use MLU_VISIBLE_DEVICES can only use"
    "share-memory only.");

namespace paddle {
namespace platform {

static int GetMLUDeviceCountImpl() {
  mluDim3 ver;
  // When cnrtDriverGetVersion is executed, the device is initialized, 
  // no longer needs to call cnrtInit().
  cnrtStatus stat = cnrtDriverGetVersion(&ver.x, &ver.y, &ver.z);
  if (stat != cnrtSuccess) {
    VLOG(2) << "MLU Driver Version can't be detected. No MLU driver!";
    return 0;
  }

  const auto *mlu_visible_devices = std::getenv("MLU_VISIBLE_DEVICES");
  if (mlu_visible_devices != nullptr) {
    std::string mlu_visible_devices_str(mlu_visible_devices);
    if (!mlu_visible_devices_str.empty()) {
      mlu_visible_devices_str.erase(
          0, mlu_visible_devices_str.find_first_not_of('\''));
      mlu_visible_devices_str.erase(
          mlu_visible_devices_str.find_last_not_of('\'') + 1);
      mlu_visible_devices_str.erase(
          0, mlu_visible_devices_str.find_first_not_of('\"'));
      mlu_visible_devices_str.erase(
          mlu_visible_devices_str.find_last_not_of('\"') + 1);
    }
    if (std::all_of(mlu_visible_devices_str.begin(),
                    mlu_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "MLU_VISIBLE_DEVICES  is set to be "
                 "empty. No MLU detected.";
      return 0;
    }
  }
  int count;
  PADDLE_ENFORCE_MLU_SUCCESS(cnDeviceGetCount(&count));
  return count;
}

int GetMLUDeviceCount() {
  static auto dev_cnt = GetMLUDeviceCountImpl();
  return dev_cnt;
}

std::vector<int> GetMLUSelectedDevices() {
  // use user specified MLUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_mlus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_mlus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetMLUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

void CheckDeviceId(int id) {
  PADDLE_ENFORCE_LT(id, GetMLUDeviceCount(),
                    platform::errors::InvalidArgument(
                        "Device id must be less than MLU count, "
                        "but received id is: %d. MLU count is: %d.",
                        id, GetMLUDeviceCount()));
}

mluDim3 GetMLUDriverVersion(int id) {
  CheckDeviceId(id);
  mluDim3 ret;
  PADDLE_ENFORCE_MLU_SUCCESS(cnGetDriverVersion(&ret.x, &ret.y, &ret.z));
  return ret;
}

int GetMLUCurrentDeviceId() {
  int device_id;
  PADDLE_ENFORCE_MLU_SUCCESS(cnrtGetDevice(&device_id));
  return device_id;
}

void SetMLUDeviceId(int id) {
  CheckDeviceId(id);
  PADDLE_RETRY_MLU_SUCCESS(cnrtSetDevice(id));
}

void GetMLUDeviceHandle(int device_ordinal, mluDeviceHandle* device) {
  cnStatus res = cnDeviceGet(device, device_ordinal);
  if (res != CN_SUCCESS) {
    VLOG(2) << "failed to get handle of MLU Device.";
  }
  PADDLE_ENFORCE_MLU_SUCCESS(res);
}

int GetMLUComputeCapability(int id) {
  CheckDeviceId(id);
  mluDeviceHandle device;
  GetMLUDeviceHandle(id, &device);

  int major, minor;
  cnStatus major_stat = 
      cnDeviceGetAttribute(&major, 
          CN_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cnStatus minor_stat = 
      cnDeviceGetAttribute(&minor,
          CN_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  PADDLE_ENFORCE_MLU_SUCCESS(major_stat);
  PADDLE_ENFORCE_MLU_SUCCESS(minor_stat);

  return major * 10 + minor;
}

}  // namespace platform
}  // namespace paddle