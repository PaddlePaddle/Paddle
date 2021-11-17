# 如何写Pass单测
总共包括4个步骤

- 1. 实现program随机生成策略函数`sample_program_config`
- 2. 实现不同推理配置函数`sample_predictor_config`
- 3. [可选]过滤非法program函数`is_program_valid`
- 4. [可选]跳过存在已知问题case函数`add_skip_pass_case`（后期需进行修复）

### sample_program_config
参考`test_fc_fuse_pass.py`中是应函数实现，罗列所有在组建program时的所有参数，采用hypothesis库生成随机数（便于Debug）。

其中在采样所有参数时，
- 1. 可以选择所有参数随机独立生成，但开发者需在is_valid_program中实现判断逻辑，使得最终的program合法
- 2. 参考`test_fc_fuse_pass.py`中逻辑，每个参数在随机时，都考虑了其它参数的影响，在合法范围内随机

### sample_predictor_config
完成各推理引擎配置，使得单测可以在cpu/gpu/mkldnn/tensorrt等多种情况下测试

### is_program_valid
用于使得最终输给单测运行的是一个正常的program

### add_skip_pass_case
添加相关存在问题的case，被判断为skip的case,即使单测出现问题，也会先忽略，不报错

## 如何复现错误case
测试过程中，可能会发现单测报错，一般日志如下
```
You can reproduce this example by temporarily adding @reproduce_failure('6.24.4', b'AAEAAQAAAAAAAAAAAAA=') as a decorator on your test case
F
======================================================================
FAIL: test (__main__.TestFcFusePass)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_fc_fuse_pass.py", line 167, in test
    self.run_and_statis(quant=False, max_examples=1000)
  File "/ssd5/jiangjiajun/inference/Paddle/python/paddle/fluid/tests/unittests/ir/inference/auto_scan_test.py", line 317, in run_and_statis
    loop_func()
  File "/ssd5/jiangjiajun/inference/Paddle/python/paddle/fluid/tests/unittests/ir/inference/auto_scan_test.py", line 311, in run_test
    return self.run_test(quant=quant, prog_config=prog_config)
  File "/ssd5/jiangjiajun/anaconda3/envs/inference/lib/python3.7/site-packages/hypothesis/core.py", line 1202, in wrapped_test
    raise the_error_hypothesis_found
  File "/ssd5/jiangjiajun/inference/Paddle/python/paddle/fluid/tests/unittests/ir/inference/auto_scan_test.py", line 311, in run_test
    return self.run_test(quant=quant, prog_config=prog_config)
  File "/ssd5/jiangjiajun/inference/Paddle/python/paddle/fluid/tests/unittests/ir/inference/auto_scan_test.py", line 395, in run_test
    self.assertTrue(status)
AssertionError: False is not true
```
将日志中的字符串`reproduce_failure('6.24.4', b'AAEAAQAAAAAAAAAAAAA=')`赋给对应单测如`test_fc_fused_pass`中调用的`self.run_and_statis`函数，再重新跑单测即可。如下所示
```
def test(self):
	self.run_and_statis(quant=False, max_examples=1000, reproduce=reproduce_failure('6.24.4', b'AAEAAQAAAAAAAAAAAAA='))
```

## 单测运行统计
单测成功跑完后，会有如下日志提示,
```
INFO:root:===================Statistical Information===================
INFO:root:Number of Generated Programs: 100
INFO:root:Number of Invalid Programs: 34
INFO:root:Number of Ran Programs: 66
INFO:root:Number of Skipped Tests: 116
.
----------------------------------------------------------------------
Ran 1 test in 5.225s

OK
```
其中，
- `Generated Programs`表示随机生成的Program总数
- `Invalid Programs`表示其中非法Program的个数，这些将被过滤，不跑测试
- `Ran Programs`表示进行了测试的Program个数
- `Skipped Tests`表示被跳过的测试次数（这些测试不管是否测试通过，都不会报错），测试次数=Program个数 * PredictorConfig个数

如若Invalid Programs个数太多，或者`Skipped Tests`太多，都表示实际测试成功的case太少，开发者可以调整产生参数的随机策略，或者加大`self.run_and_statis`中函数的`max_examples`个数
