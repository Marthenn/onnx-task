import onnx
from onnx import helper, TensorProto

def matmul_integer(size):
    input_A = helper.make_tensor_value_info('input_A', TensorProto.INT8, [1, 1, size, size])
    input_B = helper.make_tensor_value_info('input_B', TensorProto.INT8, [1, 1, size, size])
    output_Y = helper.make_tensor_value_info('output_Y', TensorProto.INT32, [1, 1, size, size])

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
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for size in sizes:
        model = matmul_integer(size)
        onnx.save(model, f'matmul_integer_{size}.onnx')
