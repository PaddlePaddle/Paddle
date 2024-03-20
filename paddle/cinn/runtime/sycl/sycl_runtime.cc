#include <paddle/cinn/runtime/sycl/sycl_runtime.h>
#include <iostream>

SYCLWorkspace* SYCLWorkspace::Global() {
  static SYCLWorkspace* inst = new SYCLWorkspace();
  return inst;
}

void SYCLWorkspace::Init(const Target::Arch arch, const std::string& platform_name) {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);

  // look for matched platform
  bool have_platform = false;
  auto platforms     = sycl::platform::get_platforms();
  std::string platform_key;
  switch (arch) {
    case Target::Arch::NVGPU:
      platform_key = "CUDA";
      break;
    case Target::Arch::AMDGPU:
      platform_key = "HIP";
      break;
    case Target::Arch::IntelGPU:
      platform_key = "Level-Zero";
      break;
    default:
      LOG(FATAL) << "SYCL Not supported arch!";
  }
  for (auto& platform : platforms) {
    std::string name = platform.get_info<sycl::info::platform::name>();
    // neither NVIDIA CUDA BACKEND nor AMD HIP BACKEND nor Intel Level-Zero
    if (name.find(platform_key) == std::string::npos) continue;
    std::vector<sycl::device> devices = platform.get_devices(sycl::info::device_type::gpu);
    this->platforms.push_back(platform);
    this->devices.insert(this->devices.end(), devices.begin(), devices.end());
    this->platform_names.push_back(platform_name);
    have_platform = true;
  }
  if (!have_platform) {
    LOG(FATAL) << "No valid gpu device/platform matched given existing options ...";
    return;
  }
  if(this->active_device_ids.size() == 0){
    // default device: 0
    std::vector<int> devicesIds = {0};
    this->SetActiveDevices(devicesIds);
  }
  initialized_ = true;
}

void SYCLWorkspace::SetActiveDevices(std::vector<int> deviceIds){
  this->active_device_ids = deviceIds;
  this->active_contexts.clear();
  this->active_queues.clear();
  this->active_events.clear();
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (const std::exception_ptr& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const sycl::exception& e) {
        std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
      }
    }
  };
  sycl::property_list q_prop{sycl::property::queue::in_order()};
  for(int deviceId : deviceIds){
    if(deviceId > this->devices.size()-1){
      LOG(FATAL) << "set valid device id! device id:" << deviceId << " > max device id:"<< this->devices.size()-1;
    }
    // create context and queue
    sycl::context* ctx = new sycl::context(this->devices[deviceId], exception_handler);
    this->active_contexts.push_back(ctx);
    // one device one queue
    sycl::queue* queue = new sycl::queue(*ctx, this->devices[deviceId], q_prop);  // In order queue
    this->active_queues.push_back(queue);
  }
  this->active_events.resize(this->active_queues.size());
  VLOG(1) << "active devices size : " << this->active_queues.size() << std::endl;
  // delete SYCLWorkspace::Global();
}

void* SYCLWorkspace::malloc(size_t nbytes, int device_id){
    void* data;
    SYCL_CALL(data = sycl::malloc_device(nbytes, *this->active_queues[device_id]))
    if(data == nullptr)
      LOG(ERROR) << "allocate sycl device memory failure!"<<std::endl;
    return data;
}

void SYCLWorkspace::free(void* data, int device_id){
  SYCL_CALL(sycl::free(data, *this->active_queues[device_id]));
}

void SYCLWorkspace::queueSync(int queue_id) {
  SYCL_CALL(this->active_queues[queue_id]->wait_and_throw());
}

void SYCLWorkspace::memcpy(void* dest, const void* src, size_t nbytes, int queue_id) {
  SYCL_CALL(this->active_queues[queue_id]->memcpy(dest, src, nbytes).wait());
}