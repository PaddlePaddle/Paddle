PR： https://github.com/PaddlePaddle/Paddle/pull/78822

1、编译Paddle
拉取Paddle PR：https://github.com/PaddlePaddle/Paddle/pull/78822
使用如下命令编译Paddle：
$ cd Paddle
$ echo /opt/dtk-25.04.2/lib >> /etc/ld.so.conf.d/dtk.conf
$ ldconfig
$ mkdir build
$ cd build
# AP依赖CINN，需设置WITH_CINN=ON；DCU需设置WITH_ROCM=ON
$ cmake .. -GNinja -DPY_VERSION=3.10 -DWITH_ROCM=ON -DWITH_CINN=ON -DWITH_DISTRIBUTE=ON
$ ninja -j16
# 安装whl包
$ pip install ./python/dist/paddlepaddle_dcu-3.5.0.dev20260421-cp310-cp310-linux_x86_64.whl


2、执行测试
将hytlass库拷贝（或软链接）到Paddle安装目录的`paddle/apy/matmul_pass/matmul/hytlass`。

`paddle/apy`是AP的文件目录，目前暂未支持自动打包到whl包中。DCU相关的AP目录结构如下：
```
paddle/apy
├── device
│   ├── cuda
│   │   └── compile_command_util.py
│   └── dcu
│       ├── compile_command_util.py
│       └── logs
│           └── test_matmul_add_relu.log
└── matmul_pass
    ├── matmul
    │   ├── ck_patch          # CK (Composable Kernel) patch，原方案，不再使用
    │   ├── cutlass_patch     # cutlass patch，CUDA侧使用
    │   ├── hytlass_patch     # hytlass patch，DCU侧使用
    │   ├── hytlass           # hytlass库（软链接，需手动创建）
    │   ├── hytlass_matmul.h  # hytlass matmul入口头文件
    │   ├── matmul.h
    │   ├── params.h
    │   └── profile.h
    ├── matmul_epilogue_pass.py
    ├── matmul_variadic_ptn.py
    └── matmul_variadic_tpl.py
```

其中`hytlass_patch`与`cutlass_patch`同级，位于Paddle仓库内，已随PR一起提交。`hytlass_patch`不是直接复用`cutlass_patch`，而是在hytlass库fork cutlass时已将patch文件中的命名做了替换（`cutlass/` → `hytlass/`、`namespace cutlass` → `namespace hytlass`、`CUTLASS_HOST_DEVICE` → `HYTLASS_HOST_DEVICE`）。

`hytlass`目录下需要手动创建软链接的关键文件：
```
hytlass
├── include
│   ├── hytlass_matmul.h
│   └── hytlass/              # hytlass库核心头文件
│       ├── epilogue/
│       ├── gemm/
│       ├── util/
│       └── ...
└── tools
    └── util
        └── include
```

创建hytlass软链接并清除缓存：

```bash
# 创建 hytlass 软链接（wheel 包中不包含符号链接）
$ ln -sf /work/hytlass /opt/py310/lib/python3.10/site-packages/paddle/apy/matmul_pass/matmul/hytlass
# 清除 axpr JSON 缓存
$ find /opt/py310/lib/python3.10/site-packages/paddle/apy/ -name "*.json" -delete
```

测试文件：`test/ap/test_matmul_add_relu.py`，测试子图为matmul + relu + add融合：
```python
def foo(
    x: pct.Tensor([B, M, K], DType),
    w: pct.Tensor([K, N], DType),
    b: pct.Tensor([B, M, N], DType),
):
    y = paddle.matmul(x, w)
    tmp = paddle.nn.functional.relu(y)
    tmp2 = tmp + b
    return tmp2
```

执行测试：

```bash
# 设置环境变量
$ export LD_LIBRARY_PATH=/opt/dtk-25.04.2/lib64:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/dcc/comgr/lib64:$LD_LIBRARY_PATH
$ export HIP_VISIBLE_DEVICES=0
$ cd /work/Paddle
$ python -m pytest test/ap/test_matmul_add_relu.py -v
```

完整的执行日志已上传到PR中。通过日志可以看到，成功生成了`pd_op.ap_variadic`算子，并使用hytlass编译了matmul kernel：
```
I0428 15:12:13.623298 35229 add_pcc_pass.cc:134] Compiling subgraph with PCC backend ...
E0428 15:12:13.623658 35229 add_pcc_pass.cc:122] 0) after ApplyApFacadePass():
{
    (%3) = "pd_op.matmul" (%0, %1) ...
    (%4) = "pd_op.relu" (%3) ...
    (%5) = "pd_op.add" (%4, %2) ...
}
...
E0428 15:12:13.889163 35229 add_pcc_pass.cc:122] 2) after ApplyApGenericDrrPass():
{
    (%6) = "builtin.combine" (%0, %1, %2) ...
    (%7) = "pd_op.ap_variadic" (%6) {
      code_module_lambda: ... "/opt/dtk-25.04.2/bin/hipcc -std=c++17 -O3 -fPIC --offload-arch=gfx928 ..."
      infer_meta_lambda: ...
    }
    (%8) = "builtin.split" (%7) ...
}
PASSED

========================= 1 passed, 1 warning in 8.94s =========================
```
