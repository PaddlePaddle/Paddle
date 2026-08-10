# fast-hadamard-transform：直接改写式迁移的历史样本

fast-hadamard-transform 是小型 CUDA extension（上游 Dao-AILab/fast-hadamard-transform，迁移分支 `paddle-migrate-fast-hadamard-transform`，ahead 1 / behind 0，单提交 10 个文件）。它是 compat 机制成熟前的迁移产物，采用的是**直接改写**而非 compat 代理：`setup.py` 直接切到 `paddle.utils.cpp_extension`、Python 接口从 `torch.autograd.Function` 整体改写为 `paddle.autograd.PyLayer`、`import torch` 全部替换为 `import paddle`。C++ 侧保留了 at::Tensor API（经 compat headers 消化），缺口处做显式桥接。今天迁移新库不应照抄这种做法——但它仍然是「compat 覆盖不到时怎么做单点 C++ 桥接」的干净样本。

### 第一落点

（历史做法，仅供对照）`setup.py` 全量改写：删除上游的预编译 wheel 下载机制与 torch 依赖，直接 `from paddle.utils.cpp_extension import CUDAExtension, CUDA_HOME, setup`；Python 接口层全部改写为 PyLayer。

### 具体例子

C++ 侧的单点桥接至今仍有参考价值：compat 缺 `torch::nn::functional::pad` 时，只补一个走 `paddle::experimental` 的等价 helper，函数签名和调用路径不动（可吸纳类别：compat 补齐 pad 入口后可还原上游写法）：

```diff
--- csrc/fast_hadamard_transform.cpp
+at::Tensor pad_last_dim(const at::Tensor &x, int64_t pad) {
+    std::vector<int> paddings(x.dim() * 2, 0);
+    paddings[paddings.size() - 1] = pad;
+    return at::Tensor(paddle::experimental::pad(x._PD_GetInner(), paddings, 0.0));
+}
 ...
     if (dim_og % 8 != 0) {
-        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
+        x = pad_last_dim(x, 8 - dim_og % 8);
     }
```

而 Python 接口层的整体改写展示了直接改写式迁移的代价——上游形状被破坏，五个 autograd Function 全部手工重写、scale 参数被迫改成类属性传递，后续 sync upstream 每次都要重做（长期负担，这正是 compat 式迁移要避免的）：

```diff
--- fast_hadamard_transform/fast_hadamard_transform_interface.py
-class HadamardTransformFn(torch.autograd.Function):
+class HadamardTransformFn(paddle.autograd.PyLayer):
     @staticmethod
-    def forward(ctx, x, scale=1.0):
-        ctx._hadamard_transform_scale = scale
-        return fast_hadamard_transform_cuda.fast_hadamard_transform(x, scale)
+    def forward(ctx, x):
+        ctx.hadamard_transform_scale = HadamardTransformFn.hadamard_transform_scale
+        _require_cuda_extension()
+        return fast_hadamard_transform_cuda.fast_hadamard_transform(
+            x, ctx.hadamard_transform_scale
+        )
```

### 优先查看的文件

- `csrc/fast_hadamard_transform.cpp`：看 `pad_last_dim`/`slice_last_dim` 单点桥接与 `AT_ERROR` → `PD_THROW` 的宏适配
- `setup.py`：看直接切换式 build（对照 FlashMLA 的 4 行 compat 开关，体会两代做法的差距）
- `fast_hadamard_transform/fast_hadamard_transform_interface.py`：看 PyLayer 整体改写的代价
- `tests/test_fast_hadamard_transform.py`：看迁移后的对拍验证

### 可复用结论

- 这是 compat 成熟前的历史做法：build 与 Python 层直接改写、只有 C++ 层走 compat headers。新迁移一律优先 compat 式最小改动
- C++ 单点桥接的三条边界（签名不变、调用路径不变、周边逻辑不变）在这个库里执行得很干净，值得复用
- 直接改写的代价在 sync upstream 时显形：上游形状破坏得越多，每次拉新要重做的就越多——用它反向校验你当前方案的补丁边界是否收敛
