import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    TestMultipleAccelerators,
)


class TestPipelineParallel(TestMultipleAccelerators):
    def test_pipeline_parallel(self):
        self.run_mnist_2accelerators('hybrid_pp_unified_dygraph_model.py')


if __name__ == "__main__":
    unittest.main()