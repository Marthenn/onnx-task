import onnx
from onnx import helper, TensorProto

def conv_integer():
    input_X = helper.make_tensor_value_info('input_X', TensorProto.INT8, [None, None])
    input_W = helper.make_tensor_value_info('input_W', TensorProto.INT8, [None, None])
    input_X_zero_point = helper.make_tensor_value_info('input_X_zero_point', TensorProto.INT8, [1])
    input_W_zero_point = helper.make_tensor_value_info('input_W_zero_point', TensorProto.INT8, [1])
    output_Y = helper.make_tensor_value_info('output_Y', TensorProto.INT32, [None, None])

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

    model = helper.make_model(graph)
    return model

if __name__ == "__main__":
    model = conv_integer()
    onnx.save(model, 'conv_integer.onnx')
