# python3.8 -m paddle.distributed.launch --gpus "0,1,2,3" demo.py 

from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
import paddle
import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.distributed.fleet as fleet


strategy = fleet.DistributedStrategy()

# 设置4路张量模型并行
model_parallel_size = 4
data_parallel_size = 1
strategy.hybrid_configs = {
   "dp_degree": data_parallel_size,
   "mp_degree": model_parallel_size,
   "pp_degree": 1
}

# 注意 strategy 是这里传递的，动态图只能这里，静态图还可以在 distributed_optimizer 里传
fleet.init(is_collective=True, strategy=strategy)

hcg = fleet.get_hybrid_communicate_group()
mp_id = hcg.get_model_parallel_rank()
rank_id = dist.get_rank()




class ExampleLayer(paddle.nn.Layer):
    def __init__(self, hidd):
        super().__init__()
        #self.fc0 = paddle.nn.Linear(hidd, 4 * hidd, bias_attr=True)
        #self.fc1 = paddle.nn.Linear(4 * hidd, hidd, bias_attr=True)

        self.fc0 = fleet.meta_parallel.ColumnParallelLinear(hidd, 4 * hidd,gather_output=False,has_bias=True,)
        self.fc1 = fleet.meta_parallel.RowParallelLinear(4 * hidd, hidd, input_is_parallel=True, has_bias=True,)

    def forward(self, x):
        x = self.fc0(x)
        x = paddle.nn.functional.gelu(x)
        x = self.fc1(x)
        dist.all_reduce(x)
        return x

x = paddle.rand([4096, 15360], dtype="float32")
mylayer = ExampleLayer(15360)


baseline_result = mylayer(x)

# 转成静态图推理
mylayer = paddle.incubate.jit.inference(mylayer) #可设置with_trt=True打开Paddle TensorRT推理
static_result = mylayer(x)


import datetime
import time

warm_up_times = 5
repeat_times = 10
paddle.device.synchronize()
starttime = datetime.datetime.now()



for i in range(100):
    static_result = mylayer(x)



paddle.device.synchronize()
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The whoel end to end time : ", time_ms / repeat_times, "ms")



print(baseline_result)
# 比较diff
print(static_result - baseline_result)


