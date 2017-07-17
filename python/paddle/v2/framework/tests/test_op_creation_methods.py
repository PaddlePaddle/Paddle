import unittest
import paddle.v2.framework.create_op_creation_methods as creation


class TestOpCreationsMethods(unittest.TestCase):
    def test_all_protos(self):
        all_protos = creation.get_all_op_protos()
        self.assertNotEqual(0, len(all_protos))

        for each in all_protos:
            self.assertTrue(each.IsInitialized())


if __name__ == "__main__":
    unittest.main()
