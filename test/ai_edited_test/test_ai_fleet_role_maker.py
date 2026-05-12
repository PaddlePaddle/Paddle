# [AUTO-GENERATED] Test file for paddle.distributed.fleet.base.role_maker
# 覆盖模块: paddle/distributed/fleet/base/role_maker.py
# 未覆盖行: 96,97,98,99,100,101,102,103,105,106,107,108,109,111,112,113,115,116,118,119,120,122,123,124,125,126,128,129,130,134,136,137,140,141,142,143,144,145,146,149
# Covered module: paddle/distributed/fleet/base/role_maker.py
# Uncovered lines: 96-149

import unittest

from paddle.distributed.fleet.base.role_maker import (
    Role,
    RoleMakerBase,
    UserDefinedRoleMaker,
)


class TestRole(unittest.TestCase):
    """测试 Role 枚举类
    Test Role enum class"""

    def test_role_values(self):
        """测试 Role 枚举值
        Test Role enum values"""
        self.assertIsNotNone(Role.WORKER)
        self.assertIsNotNone(Role.SERVER)


class TestRoleMakerBase(unittest.TestCase):
    """测试 RoleMakerBase 类
    Test RoleMakerBase class"""

    def test_role_maker_base_import(self):
        """测试 RoleMakerBase 可以被导入
        Test RoleMakerBase can be imported"""
        self.assertIsNotNone(RoleMakerBase)


class TestUserDefinedRoleMaker(unittest.TestCase):
    """测试 UserDefinedRoleMaker 类
    Test UserDefinedRoleMaker class"""

    def test_user_defined_role_maker_import(self):
        """测试 UserDefinedRoleMaker 可以被导入
        Test UserDefinedRoleMaker can be imported"""
        self.assertIsNotNone(UserDefinedRoleMaker)


if __name__ == '__main__':
    unittest.main()
