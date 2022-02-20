# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections

'''
------------------------->     Profiling Report     <-------------------------
(Only list devices used by current program)
Device                    Utilization (%)                             
 CPU                          xxx
 GPU0                         xxx
 GPU1                         xxx 

-------------------------     Overview Summary      -------------------------
Total time: 840.473
                               Time            Ratio
  DataLoader                   816.12           2%
  Operator                   Pro  416.12           50%
    prepare_data
    Infer shape 
    Computation                816.12           96%  
      Kernel                   400.12         (Async)
  Framework overhead           24.3528          2%

-------------------------     Model Summary         -------------------------   
                              Time                Ratio
  DataLoader
   Forward
   Backward
  Optimization
    Others
-------------------------     Distributed Summary   -------------------------
-------------------------      Operator Summary      -------------------------
Event                                           Calls    CPU Time     GPU Time
bilinear_interp                                 200      32.105901    44.445341
    prepare_data                                200      32.105901    44.445341
    infer_shape                                 200      32.105901    44.445341
    compute                                     200      32.105901    44.445341
    xxxxxxx_kernel                              200      32.105901    44.445341
bilinear_interp_grad                            200      32.105901    44.445341
    prepare_data                                200      32.105901    44.445341
    infer_shape                                 200      32.105901    44.445341
    compute                                     200      32.105901    44.445341
    xxxxxxx_kernel                              200      32.105901    44.445341
conv2d                                          200      32.105901    44.445341
    prepare_data                                200      32.105901    44.445341
    infer_shape                                 200      32.105901    44.445341
    compute                                     200      32.105901    44.445341
    xxxxxxx_kernel                              200      32.105901    44.445341
conv2d_grad                                     200      32.105901    44.445341
    prepare_data                                200      32.105901    44.445341
    infer_shape                                 200      32.105901    44.445341
    compute                                     200      32.105901    44.445341
    xxxxxxx_kernel                              200      32.105901    44.445341
'''


_AllTracerEventType = [TracerEventType.Dataloader, TracerEventType.ProfileStep, 
                      TracerEventType.CudaRuntime, TracerEventType.Kernel,  TracerEventType.Memcpy,
                      TracerEventType.Memset, TracerEventType.UserDefined, TracerEventType.OperatorInner,
                      TracerEventType.Forward, TracerEventType.Backward, TracerEventType.Optimization,
                      TracerEventType.Communication, TracerEventType.PythonOp]

class TimeRangeSummary:
  def __init__(self):
    self.CPUTimeRange = collections.defaultdict(list)
    self.GPUTimeRange = collections.defaultdict(lambda: collections.defaultdict(list)) // GPU events should be divided into different devices
    self.CPUTimeRangeSum = collections.defaultdict(int)
    self.GPUTimeRangeSum = collections.defaultdict(lambda: collections.defaultdict(int))
  def parse(self, nodetrees):
    '''
    Analysis node trees in profiler result, and get time range for different tracer event type
    '''
    pass

  def get_gpu_devices(self):
    return self.GPUTimeRangeSum.keys()
  
  def get_range_sum(self, device, type):
    pass


class OperatorSummary:
  def __init__(self):
    pass

class EventAnalyseNode:
  """
  Node used for analysing statistic data in node tree.
  """
  def __init__(self, node):
    self.node = node

  def __getattr__(self, name)
    return getattr(self.node, name)

  
