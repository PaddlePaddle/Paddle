import paddle.v2.framework.core as core
import unittest
import numpy as np


class TestSelectedRows(unittest.TestCase):
    def test_selected_rows(self):
        place = core.CPUPlace()
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        selected_rows = core.SelectedRows(rows, height)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0
        tensor = selected_rows.get_tensor()
        tensor.set(np_array, place)

        # compare rows
        self.assertEqual(0, selected_rows.rows()[0])
        self.assertEqual(4, selected_rows.rows()[1])
        self.assertEqual(7, selected_rows.rows()[2])

        # compare height
        self.assertEqual(10, selected_rows.height())

        # compare tensor
        self.assertAlmostEqual(2.0,
                               selected_rows.get_tensor().get_float_element(0))
        self.assertAlmostEqual(1.0,
                               selected_rows.get_tensor().get_float_element(1))
        self.assertAlmostEqual(
            4.0,
            selected_rows.get_tensor().get_float_element(2 * row_numel + 8))


if __name__ == "__main__":
    unittest.main()
