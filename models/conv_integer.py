import onnx
from onnx import helper, TensorProto

def conv_integer(size):
    input_X = helper.make_tensor_value_info('input_X', TensorProto.UINT8, [1, 1, size, size])
    input_W = helper.make_tensor_value_info('input_W', TensorProto.UINT8, [1, 1, size, size])
    input_X_zero_point = helper.make_tensor_value_info('input_X_zero_point', TensorProto.UINT8, [1])
    input_W_zero_point = helper.make_tensor_value_info('input_W_zero_point', TensorProto.UINT8, [1])
    output_Y = helper.make_tensor_value_info('output_Y', TensorProto.INT32, [1, 1, 1, 1])

    conv_node = helper.make_node(
        'ConvInteger',
        inputs=['input_X', 'input_W', 'input_X_zero_point', 'input_W_zero_point'],
        outputs=['output_Y']
    )

    graph = helper.make_graph(
        [conv_node],
        'conv_integer',
        [input_X, input_W, input_X_zero_point, input_W_zero_point],
        [output_Y]
    )

    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

if __name__ == "__main__":
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for size in sizes:
        model = conv_integer(size)
        onnx.save(model, f'conv_integer_{size}.onnx')
