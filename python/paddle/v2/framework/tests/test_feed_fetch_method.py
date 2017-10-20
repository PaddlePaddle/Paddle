import paddle.v2.framework.core as core
import unittest
import numpy as np


class TestFeedFetch(unittest.TestCase):
    def test_feed_fetch(self):
        scope = core.Scope()
        place = core.CPUPlace()
        input_array = np.ones((4, 4, 6)).astype("float32")
        input_array[0, 0, 0] = 3
        input_array[3, 3, 5] = 10
        input_tensor = core.LoDTensor([[0, 2, 4]])
        input_tensor.set(input_array, place)

        core.set_feed_variable(scope, input_tensor, "feed", 0)

        output_tensor = core.get_fetch_variable(scope, "feed", 0)

        output_lod = output_tensor.lod()
        self.assertEqual(0, output_lod[0][0])
        self.assertEqual(2, output_lod[0][1])
        self.assertEqual(4, output_lod[0][2])

        output_array = np.array(output_tensor)
        self.assertEqual(3, output_array[0, 0, 0])
        self.assertEqual(10, output_array[3, 3, 5])


if __name__ == "__main__":
    unittest.main()
