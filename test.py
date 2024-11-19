import unittest

import numpy as np
import onnxruntime as ort

class MyTestCase(unittest.TestCase):
    def test_matmul(self):
        mat_A = np.random.randint(-128, 127, (3, 3)).astype(np.int8)
        mat_B = np.random.randint(-128, 127, (3, 3)).astype(np.int8)

        model_path = 'matmul_integer.onnx'
        sess = ort.InferenceSession(model_path)
        inputs = {
            'input_A': mat_A,
            'input_B': mat_B
        }
        output = sess.run(None, inputs)
        np_res = np.matmul(mat_A.astype(np.int32), mat_B.astype(np.int32))
        np.testing.assert_array_equal(np_res, output[0])

    def test_conv(self):
        # Issue with ConvInteger in onnxruntime
        pass

if __name__ == '__main__':
    unittest.main()
