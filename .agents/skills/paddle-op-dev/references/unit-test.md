# 单元测试开发

## 目录
- [概述](#概述)
- [测试文件位置](#测试文件位置)
- [测试框架 OpTest](#测试框架-optest)
- [前向测试](#前向测试)
- [反向测试](#反向测试)
- [完整测试示例](#完整测试示例)
- [测试运行方式](#测试运行方式)
- [注意事项](#注意事项)

## 概述

新增算子需要在 `test/legacy_test` 目录下添加单元测试，使用 `OpTest` 框架验证算子前向计算和反向梯度的正确性。

## 测试文件位置

```
test/legacy_test/test_xxx_op.py
```

## 测试框架 OpTest

继承 `OpTest` 基类来编写算子测试。

```python
from paddle.base import core
from paddle.base.tests.unittests.op_test import OpTest
import unittest
import numpy as np
```

### OpTest 基本结构

```python
class TestXxxOp(OpTest):
    def setUp(self):
        self.op_type = "xxx"           # 算子名
        self.python_api = paddle.xxx   # 对应的 Python API
        self.init_config()
        # 设置输入
        self.inputs = {
            'X': np.random.random((3, 4)).astype("float64"),
        }
        # 设置属性
        self.attrs = {
            'attr1': value1,
        }
        # 设置期望输出
        self.outputs = {
            'Out': expected_output,
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)
```

## 前向测试

`check_output()` 验证 C++ 算子前向计算结果与 Python/NumPy 参考实现的一致性。

```python
def test_check_output(self):
    self.check_output(check_pir=True)
```

- `check_pir=True`：同时检验 PIR 模式下的输出正确性

## 反向测试

`check_grad()` 验证反向梯度计算的正确性（通过数值微分法）。

```python
def test_check_grad(self):
    self.check_grad(
        ['X'],           # 需要检查梯度的输入列表
        'Out',           # 对应的输出名
        check_pir=True
    )
```

- 输入数据建议使用 `float64` 类型以保证数值微分精度
- 如果某些输入不需要检查梯度，可使用 `no_grad_set` 参数

```python
def test_check_grad_no_x(self):
    self.check_grad(
        ['Y'], 'Out',
        no_grad_set=set("X"),
        check_pir=True
    )
```

## 完整测试示例

以 trace 算子为例（`test/legacy_test/test_trace_op.py`）：

```python
import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.base.tests.unittests.op_test import OpTest


class TestTraceOp(OpTest):
    def setUp(self):
        self.op_type = "trace"
        self.python_api = paddle.trace
        self.init_config()

    def init_config(self):
        self.case = np.random.randn(20, 6).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.outputs = {
            'Out': np.trace(self.inputs['Input'],
                           offset=self.attrs['offset'],
                           axis1=self.attrs['axis1'],
                           axis2=self.attrs['axis2'])
        }

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['Input'], 'Out', check_pir=True)


class TestTraceOpCase1(TestTraceOp):
    """测试不同参数配置"""
    def init_config(self):
        self.case = np.random.randn(20, 6).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 1, 'axis1': 0, 'axis2': 1}
        self.outputs = {
            'Out': np.trace(self.inputs['Input'],
                           offset=self.attrs['offset'],
                           axis1=self.attrs['axis1'],
                           axis2=self.attrs['axis2'])
        }


class TestTraceOpCase2(TestTraceOp):
    """测试高维输入"""
    def init_config(self):
        self.case = np.random.randn(2, 20, 2, 3).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 1, 'axis1': 1, 'axis2': 2}
        self.outputs = {
            'Out': np.trace(self.inputs['Input'],
                           offset=self.attrs['offset'],
                           axis1=self.attrs['axis1'],
                           axis2=self.attrs['axis2'])
        }


class TestTraceAPICase(unittest.TestCase):
    """测试 Python API 调用"""
    def test_case1(self):
        x = paddle.to_tensor(np.random.randn(20, 6).astype('float64'))
        result = paddle.trace(x)
        expected = np.trace(x.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
```

## 测试运行方式

```bash
# 单个测试文件
python -m pytest test/legacy_test/test_xxx_op.py -v

# 或者
python test/legacy_test/test_xxx_op.py
```

## 注意事项

1. **数据类型**：反向梯度检查使用 `float64` 确保数值微分精度
2. **多 Case 覆盖**：继承基类并重写 `init_config()` 测试不同参数组合
3. **边界条件**：测试边界情况（零维、高维、负轴索引等）
4. **Python API 测试**：除了 OpTest 还需要测试 Python API 直接调用
5. **check_pir=True**：新增算子测试中需添加此参数
6. **op_type**：须与 YAML 配置中的算子名一致
7. **python_api**：须设置为对应的 Python API 函数
