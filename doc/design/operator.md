# Operator Design

Operator in PaddlePaddle mainly describe how to do operation with Variable. It does not actually contains any data or state, but with reference of these Variables/State.

Op will get/update data/state from Scope when running.

Operator have a template parameter `DeviceContext`. DeviceContext is used to specify on which device this op will run. Each Op will implement multi Op according to different DeviceContext.

```cpp
#pragma once

namespace paddle {
namespace framework {

template <class DeviceContext>
class Operator {
 public:
  explicit Operator(const OperatorDef& operator_def, Scope* sc)
          : device_context_(operator_def.device_config()) {
      device_context_.SwitchToDevice(0);
    }
  ~Operator() {}

  // Run function of Operator is used to switch the device context,
  // actual computation is in RunOnDevice().
  bool Run(int stream_id = 0, Scope* scope) final {
    return RunOnDevice();
  }

  // Actual computation implement in RunOnDevice.
  virtual bool RunOnDevice() = 0;

protected:
  // Context is a template class variable to mark and managing where to run the operator.
  DeviceContext device_context_;
  vector<string*> inputs_;
  vector<string*> outputs_;
};
}  // namespace framework
}  // namespace paddle
```

