import onnx
from onnx import helper, TensorProto

def matmul_integer():
    input_A = helper.make_tensor_value_info('input_A', TensorProto.INT8, [1, 1, 2, 2])
    input_B = helper.make_tensor_value_info('input_B', TensorProto.INT8, [1, 1, 2, 2])
    output_Y = helper.make_tensor_value_info('output_Y', TensorProto.INT32, [1, 1, 2, 2])

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

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    return model

if __name__ == "__main__":
    model = matmul_integer()
    onnx.save(model, 'models/matmul_integer.onnx')
