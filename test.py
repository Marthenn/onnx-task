import unittest

import numpy as np
import onnxruntime as ort

class MyTestCase(unittest.TestCase):
    def test_matmul(self):
        mat_A = np.random.randint(-128, 127, (1, 1, 2, 2)).astype(np.int8)
        mat_B = np.random.randint(-128, 127, (1, 1, 2, 2)).astype(np.int8)

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
        mat_X = np.random.randint(0, 225, (1, 1, 2, 2)).astype(np.uint8)
        mat_W = np.random.randint(0, 225, (1, 1, 2, 2)).astype(np.uint8)
        zero_X = np.random.randint(0, 225, (1)).astype(np.uint8)
        zero_W = np.random.randint(0, 225, (1)).astype(np.uint8)

        model_path = 'conv_integer.onnx'
        sess = ort.InferenceSession(model_path)
        inputs = {
            'input_X': mat_X,
            'input_W': mat_W,
            'input_X_zero_point': zero_X,
            'input_W_zero_point': zero_W
        }
        output = sess.run(None, inputs)

        mat_X = mat_X.astype(np.int32)
        mat_W = mat_W.astype(np.int32)
        zero_X = zero_X.astype(np.int32)
        zero_W = zero_W.astype(np.int32)

        mat_X = mat_X - zero_X
        mat_W = mat_W - zero_W
        np_res = np.zeros((1, 1, 1, 1), dtype=np.int32)
        for i in range(2):
            for j in range(2):
                np_res[0, 0, 0, 0] += mat_X[0, 0, i, j] * mat_W[0, 0, i, j]

        np.testing.assert_array_equal(np_res, output[0])

if __name__ == '__main__':
    unittest.main()
