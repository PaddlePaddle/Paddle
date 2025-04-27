import os
import sys
import paddle
from paddle.base.core import CUDAPlace, CustomPlace

# 检查是否启用了自定义设备
CUSTOM_DEVICE = os.environ.get("CUSTOM_DEVICE", None)
if not CUSTOM_DEVICE:
    # 未启用自定义设备，直接退出
    sys.exit(0)

# ----------------------------
# 核心逻辑：全局替换所有 CUDAPlace 和 'gpu'
# ----------------------------

# 1. 替换所有 CUDAPlace 的实例化
def _new_cuda_place(*args, ​**​kwargs):
    """将 CUDAPlace 的实例化替换为 CustomPlace"""
    return CustomPlace(CUSTOM_DEVICE, *args, ​**​kwargs)

# 猴子补丁：覆盖 paddle.base.core.CUDAPlace
paddle.base.core.CUDAPlace = _new_cuda_place
# 覆盖其他可能的导入路径（如 from paddle import CUDAPlace）
paddle.CUDAPlace = _new_cuda_place

# 2. 替换所有字符串 'gpu' 为自定义设备名
def _patch_string_gpu(module):
    """递归遍历模块，替换代码中的 'gpu' 字符串"""
    import ast

    class StringReplaceVisitor(ast.NodeTransformer):
        def visit_Str(self, node):
            if node.s == 'gpu':
                return ast.Str(s=CUSTOM_DEVICE)
            return node

    # 遍历模块的 AST 并修改
    tree = ast.parse(module.__dict__['__patched_source__'])
    visitor = StringReplaceVisitor()
    new_tree = visitor.visit(tree)
    ast.fix_missing_locations(new_tree)

    # 重新编译并替换模块代码
    exec(compile(new_tree, filename="<ast>", mode="exec"), module.__dict__)

# 3. 自动跳过原 CUDA 相关条件检查
def _patch_skipif_decorators(module):
    """修改装饰器中的条件检查（如 @unittest.skipIf）"""
    import ast

    class SkipIfVisitor(ast.NodeTransformer):
        def visit_Call(self, node):
            # 匹配类似 `core.is_compiled_with_cuda()` 的条件
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == 'is_compiled_with_cuda'
            ):
                # 替换为 True 或 False（根据需求调整）
                return ast.NameConstant(value=False)
            return node

    tree = ast.parse(module.__dict__['__patched_source__'])
    visitor = SkipIfVisitor()
    new_tree = visitor.visit(tree)
    ast.fix_missing_locations(new_tree)
    exec(compile(new_tree, filename="<ast>", mode="exec"), module.__dict__)

# 4. 动态打补丁到所有已加载模块
import importlib

def _monkey_patch_all_modules():
    """遍历所有已加载模块并应用补丁"""
    for module_name in list(sys.modules.keys()):
        module = sys.modules[module_name]
        if not hasattr(module, '__file__') or 'site-packages' in module.__file__:
            continue  # 跳过第三方库

        # 获取模块源代码
        try:
            with open(module.__file__, 'r') as f:
                source = f.read()
        except (TypeError, OSError):
            continue

        # 保存原始代码并应用 AST 修改
        module.__patched_source__ = source
        _patch_string_gpu(module)
        _patch_skipif_decorators(module)

# 执行全局补丁
_monkey_patch_all_modules()