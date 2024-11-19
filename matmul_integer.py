import onnx
from onnx import helper, TensorProto

if __name__ == "__main__":
    input_A = helper.make_tensor_value_info('input_A', TensorProto.INT8, [None, None])
    input_B = helper.make_tensor_value_info('input_B', TensorProto.INT8, [None, None])
    output_Y = helper.make_tensor_value_info('output_Y', TensorProto.INT32, [None, None])

    matmul_node = helper.make_node(
        'MatMulInteger',
        inputs=['input_A', 'input_B'],
        outputs=['output_Y']
    )

    graph = helper.make_graph(
        [matmul_node],
        'matmul_integer',
        [input_A, input_B],
        [output_Y]
    )

    model = helper.make_model(graph)
    onnx.save(model, 'matmul_integer.onnx')
