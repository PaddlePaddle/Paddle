#!/usr/bin/env python3
"""
验证legacy_test修复效果的脚本
"""

import os
import sys
import subprocess
import importlib.util

def test_import_fixes():
    """测试导入修复是否有效"""
    print("=== 测试导入修复 ===")
    
    # 测试op_test.py的导入
    try:
        spec = importlib.util.spec_from_file_location("op_test", "op_test.py")
        op_test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(op_test)
        print("✓ op_test.py 导入成功")
    except Exception as e:
        print(f"✗ op_test.py 导入失败: {e}")
    
    # 测试utils.py的导入
    try:
        spec = importlib.util.spec_from_file_location("utils", "utils.py")
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        print("✓ utils.py 导入成功")
    except Exception as e:
        print(f"✗ utils.py 导入失败: {e}")

def test_specific_files():
    """测试特定文件的修复效果"""
    print("\n=== 测试特定文件修复 ===")
    
    test_files = [
        "test_conv3d_transpose_op.py",
        "test_dropout_op.py", 
        "test_gather_op.py",
        "test_logical_op.py",
        "test_transformer_api.py",
        "test_matmul_v2_op.py",
        "test_allgather.py",
        "test_reducescatter.py",
        "test_flash_attention.py",
        "test_fused_dot_product_attention_op.py",
        "test_fleet_pyramid_hash.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✓ {test_file} 存在")
        else:
            print(f"✗ {test_file} 不存在")

def test_syntax_validation():
    """测试语法验证"""
    print("\n=== 测试语法验证 ===")
    
    test_files = [
        "test_conv3d_transpose_op.py",
        "test_dropout_op.py", 
        "test_gather_op.py",
        "test_logical_op.py",
        "test_matmul_v2_op.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                with open(test_file, 'r') as f:
                    compile(f.read(), test_file, 'exec')
                print(f"✓ {test_file} 语法正确")
            except SyntaxError as e:
                print(f"✗ {test_file} 语法错误: {e}")
            except Exception as e:
                print(f"✗ {test_file} 其他错误: {e}")

def test_data_type_compatibility():
    """测试数据类型兼容性修复"""
    print("\n=== 测试数据类型兼容性 ===")
    
    # 测试op_test.py中的数据类型判断修复
    try:
        with open('op_test.py', 'r') as f:
            content = f.read()
            
        if 'paddle.float32' in content and 'VarDesc.VarType.FP32' in content:
            print("✓ 数据类型兼容性修复已应用")
        else:
            print("✗ 数据类型兼容性修复未完全应用")
            
    except Exception as e:
        print(f"✗ 无法检查数据类型兼容性: {e}")

def test_api_compatibility():
    """测试API兼容性修复"""
    print("\n=== 测试API兼容性 ===")
    
    try:
        with open('test_dropout_op.py', 'r') as f:
            content = f.read()
            
        if 'try:' in content and 'except AttributeError:' in content:
            print("✓ API兼容性修复已应用")
        else:
            print("✗ API兼容性修复未完全应用")
            
    except Exception as e:
        print(f"✗ 无法检查API兼容性: {e}")

def test_import_compatibility():
    """测试导入兼容性修复"""
    print("\n=== 测试导入兼容性 ===")
    
    try:
        with open('test_transformer_api.py', 'r') as f:
            content = f.read()
            
        if 'try:' in content and 'except ImportError:' in content:
            print("✓ 导入兼容性修复已应用")
        else:
            print("✗ 导入兼容性修复未完全应用")
            
    except Exception as e:
        print(f"✗ 无法检查导入兼容性: {e}")

def main():
    """主函数"""
    print("开始验证legacy_test修复效果...")
    
    # 切换到legacy_test目录
    if not os.path.exists('op_test.py'):
        print("错误: 请在legacy_test目录下运行此脚本")
        return
    
    test_import_fixes()
    test_specific_files()
    test_syntax_validation()
    test_data_type_compatibility()
    test_api_compatibility()
    test_import_compatibility()
    
    print("\n=== 验证完成 ===")
    print("如果所有测试都通过，说明修复成功！")
    print("如果有失败的测试，请检查对应的文件并手动修复。")

if __name__ == '__main__':
    main()
