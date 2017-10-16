import paddle.v2.framework.core as core
import unittest
import numpy as np

# class TestFeedFetch(unittest.TestCase):
# 	def test_feed_fetch(self):
# 		place = core.CPUPlace()
# 		input_tensor = core.LoDTensor([[0, 2, 4]])
# 		input_tensor.set_dims([4, 4, 6])
# 		input_tensor.alloc_int(place)
# 		input_array = np.array(input_tensor)
# 		input_array[0, 0, 0] = 3
# 		input_array[3, 3, 5] = 10
# 		input_tensor.set(input_array, place)

# core.set_feed_variable(input_tensor, "feed", 0)

# output_tensor = core.get_fetch_variable("feed", 0)
# print type(output_tensor)

# output_lod = output_tensor.lod()
# print type(output_lod)
# print output_lod[0]
# print output_lod[0][0]
# print output_lod[0][1]
# print output_lod[0][2]
# # self.assertEqual(0, output_lod[0][0])
# # self.assertEqual(0, output_lod[0][0])
#       # self.assertEqual(2, output_lod[0][1])
#       # self.assertEqual(4, output_lod[0][2])

#       # output_array = np.array(output_tensor)
#       # self.assertEqual(3, output_array[0, 0, 0])
#       # self.assertEqual(10, output_array[3, 3, 5]);


class TestFeedFetch(unittest.TestCase):
    def test_feed_fetch(self):
        place = core.CPUPlace()
        input_tensor = core.LoDTensor([[0, 2, 4]])
        input_tensor.set_dims([4, 4, 6])
        input_tensor.alloc_float(place)
        input_array = np.array(input_tensor)
        input_array[0, 0, 0] = 3
        input_array[3, 3, 5] = 10
        input_tensor.set(input_array, place)


if __name__ == "__main__":
    unittest.main()
