from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
import onnx.numpy_helper as numpy_helper
import numpy as np
import torch
import onnx

from onnx.shape_inference import infer_shapes
from inject_utils.layers import perturb_quantizer
from inject_utils.layers import float32_bit_flip
from inject_utils.layers import delta_init
from inject_utils.layers import int_bit_flip

import time
import copy

def execute_node(node, main_graph, final_output_node, weight_dict, module, inject_parameters=None):
    node_inputs = []
    node_outputs = []

    added_quant_inputs, added_quant_outputs, list_operation_time = expand_node_inputs_outputs(main_graph, node, weight_dict, module)
    node_inputs += added_quant_inputs
    node_outputs += added_quant_outputs

    desired_node_outputs = [x for x in node_outputs if x.name == final_output_node]
    intermediate_node_outputs = [x for x in node_outputs if x.name != final_output_node]
    
    for index, node_input in enumerate(node.input):
        if (len(node_input)) == 0:
            node.input[index] = node_inputs[-1].name

    graph = helper.make_graph(
            nodes = [node],
            name = "single_node_exec",
            inputs = node_inputs,
            outputs = desired_node_outputs
    )

    model = ModelProto()
    model = infer_shapes(model)
    model.graph.CopyFrom(graph)
    model.opset_import.append(OperatorSetIdProto(version=13))
    model = ModelWrapper(model)

    input_dict = {}
    for node_iter in node_inputs:
        if node_iter.name == [node_intermediate.name for node_intermediate in intermediate_node_outputs]:
            continue
        if node_iter.name in [node_intermediate.name for node_intermediate in node_outputs]:
            continue
        input_dict[node_iter.name] = weight_dict[node_iter.name]

    output_tensors = execute_onnx(model, input_dict)
    tensor_output_name = list(output_tensors.keys())[0]
    original_tensor_output = output_tensors[tensor_output_name]
    print("OUTPUT Y:")
    print(original_tensor_output)
    weight_dict[tensor_output_name] = output_tensors[tensor_output_name]

    if inject_parameters and ("RANDOM" in inject_parameters["inject_type"]) and (node.op_type == inject_parameters["faulty_operation_name"]):
        print("FOUND HERE RANDOM:")
        print(node.name)
        faulty_value = None
        target_indices = [np.random.randint(0, dim) for dim in weight_dict[tensor_output_name].shape]
        golden_value = weight_dict[tensor_output_name][tuple(target_indices)]
        print(weight_dict[tensor_output_name][tuple(target_indices)])
        if "BITFLIP" in inject_parameters["inject_type"]:
            faulty_value, flip_bit = float32_bit_flip(weight_dict[tensor_output_name], target_indices)
        else:
            faulty_value = delta_init()
        weight_dict[tensor_output_name][tuple(target_indices)] = faulty_value
        print("FAULTY:")
        print(faulty_value)

    if inject_parameters and (inject_parameters["inject_type"] in ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16"]):
        # First layer in faulty_trace, obtains perturbations
        if inject_parameters["faulty_tensor_name"] in node.input:
            faulty_value, target_indices = int_bit_flip(weight_dict, inject_parameters["faulty_tensor_name"], inject_parameters["faulty_bit_position"], 4)
            weight_dict["delta_4d"] = np.zeros_like(weight_dict[inject_parameters["faulty_tensor_name"]])
            weight_dict["delta_4d"][tuple(target_indices)] = faulty_value
            perturb = weight_dict["delta_4d"][tuple(target_indices)] - weight_dict[inject_parameters["faulty_tensor_name"]][tuple(target_indices)]
            weight_dict["delta_4d"][tuple(target_indices)] = perturb

            #TODO: fix this
            input_dict_original = input_dict.copy()
            intermediate_input_name = inject_parameters["faulty_tensor_name"]
            for input_node in node.input:
                if input_node == inject_parameters["faulty_tensor_name"]:
                    intermediate_input_name = input_node
            input_dict[intermediate_input_name] = weight_dict["delta_4d"]
            intermediate_output_tensors = execute_onnx(model, input_dict)
            print(input_dict)
            weight_dict["delta_4d"] = intermediate_output_tensors[list(intermediate_output_tensors)[0]]
            input_dict = input_dict_original

        # Final layer in faulty_trace, should be the target layer and applies the fault models
        faulty_operation = inject_parameters["faulty_operation_name"]
        if faulty_operation == inject_parameters["faulty_operation_name"]:
            print("FINAL LAYER")
            print(faulty_operation)
            if "INPUT16" == inject_parameters["inject_type"]:
                delta_16 = np.zeros(weight_dict["delta_4d"].shape, dtype=np.float32)
                random_shape = list(weight_dict["delta_4d"].shape)
                row_index = random_shape[3]//16
                if row_index == 0:
                    row_index = 0
                else:
                    row_index = np.random.randint(0, row_index)
                row_index = row_index*16
                indices = []
                if len(np.nonzero(weight_dict["delta_4d"])[0]) > 0:
                    for shape_index_array in np.nonzero(weight_dict["delta_4d"]):
                        indices.append(list(shape_index_array)[0])
                    indices[3] = row_index

                    for i in range(16):
                        if i >= random_shape[3]:
                            break
                        delta_16[tuple(indices)] = weight_dict["delta_4d"][(tuple(indices))]
                        indices[3] = indices[3] + 1
                    weight_dict["delta_4d"] = delta_16

            elif "WEIGHT16" == inject_parameters["inject_type"]:
                delta_16 = np.zeros(weight_dict["delta_4d"].shape, dtype=np.float32)
                random_shape = list(weight_dict["delta_4d"].shape)
                column_index = random_shape[2]//16
                if column_index == 0:
                    column_index = 0
                else:
                    column_index = np.random.randint(0, column_index)
                column_index = column_index*16
                indices = []
                if len(np.nonzero(weight_dict["delta_4d"])[0]) > 0:
                    for shape_index_array in np.nonzero(weight_dict["delta_4d"]):
                        indices.append(list(shape_index_array)[0])
                    indices[2] = column_index

                    for i in range(np.random.randint(1,16)):
                        if i >= random_shape[2]:
                            break
                        delta_16[tuple(indices)] = weight_dict["delta_4d"][(tuple(indices))]
                        indices[2] = indices[2] + 1
                    weight_dict["delta_4d"] = delta_16
            else:
                print("INPUTS/WEIGHTS")
            print("FAULT INJECTED!")
            print(np.nonzero(weight_dict["delta_4d"]))
            print("After nonzero")
            print(weight_dict["delta_4d"])
            temp_variable = (np.add(weight_dict[tensor_output_name], weight_dict["delta_4d"]))
            weight_dict[tensor_output_name] = temp_variable
            output_tensors[tensor_output_name] = temp_variable

    return output_tensors, weight_dict, list_operation_time

def inference(main_graph, weight_dict, module, inject_parameters=None):
    def execute_single_node(node, weight_dict, main_graph, module):
        final_output_node = node.output[0]
        output_tensors, weight_dict, list_operation_time = execute_node(node, main_graph, final_output_node, weight_dict, module, inject_parameters)
        return output_tensors, weight_dict, list_operation_time
    output_tensors = None
    for node in main_graph.node:
        start_time = time.time()
        output_tensors, weight_dict, list_operation_time = execute_single_node(node, weight_dict, main_graph, module)
    print("OUTPUT Y FAULTY:")
    print(output_tensors[list(output_tensors.keys())[0]])
    return output_tensors, weight_dict

def expand_node_inputs_outputs(graph, node, weight_dict, module):
    added_inputs = []
    added_outputs = []

    added_inputs += list(filter(lambda x: x.name in node.input, graph.input))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.output))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))

    start_time = time.time()
    return added_inputs, added_outputs, time.time() - start_time

def get_weight_dict(module_path):
    module = ModelWrapper(module_path)
    module_graph = module.graph
    module_weights = module.graph.initializer
    module_weight_dict = {}
    for weight in module_weights:
        module_weight_dict[weight.name] = numpy_helper.to_array(weight)
    return module_graph, module_weight_dict

def prepare_inference(module_path, module_input_values):
    module = ModelWrapper(module_path)
    output = [node.name for node in module.graph.output]

    input_all = [node.name for node in module.graph.input]
    input_initializers = [node.name for node in module.graph.initializer]
    module_input_names = list(set(input_all) - set(input_initializers))

    module_graph, module_weight_dict = get_weight_dict(module_path)

    for input_name in module_input_names:
        module_weight_dict[input_name] = module_input_values[input_name]

    return module_weight_dict, module_graph

def run_module(module, input_values, module_filepath, inject_parameters=None):
    module_weight_dict, module_graph = prepare_inference(module_filepath, input_values)
    for input_name in list(input_values.keys()):
        module_weight_dict[input_name] = input_values[input_name]
    return inference(module_graph, module_weight_dict, module, inject_parameters)

if __name__ == "__main__":
    # Matmul Integer Injection
    # module_filepath = "models/matmul_integer.onnx"
    # mat_A = np.random.randint(-128, 127, (1, 1, 2, 2)).astype(np.int8)
    # mat_B = np.random.randint(-128, 127, (1, 1, 2, 2)).astype(np.int8)
    # input_values = {
    #     "input_A": mat_A,
    #     "input_B": mat_B
    # }
    # inject_parameters = {}
    # inject_parameters["inject_type"] = "INPUT"
    # inject_parameters["faulty_tensor_name"] = "input_A"
    # inject_parameters["faulty_bit_position"] = 1
    # inject_parameters["faulty_output_tensor"] = "output_Y"
    # inject_parameters["faulty_operation_name"] = "MatMulInteger"
    # print(run_module(None, input_values, module_filepath, inject_parameters))

    # Conv Integer Injection
    module_filepath = "models/conv_integer.onnx"
    mat_X = np.random.randint(0, 225, (1, 1, 3, 3)).astype(np.uint8)
    mat_W = np.random.randint(0, 225, (1, 1, 2, 2)).astype(np.uint8)
    zp_X = np.random.randint(0, 225, (1)).astype(np.uint8)
    zp_W = np.random.randint(0, 225, (1)).astype(np.uint8)
    input_values = {
        "input_X": mat_X,
        "input_W": mat_W,
        "input_X_zero_point": zp_X,
        "input_W_zero_point": zp_W
    }
    inject_parameters = {}
    inject_parameters["inject_type"] = "WEIGHT"
    inject_parameters["faulty_tensor_name"] = "input_X"
    inject_parameters["faulty_bit_position"] = 0
    inject_parameters["faulty_output_tensor"] = "output_Y"
    inject_parameters["faulty_operation_name"] = "ConvInteger"
    print(run_module(None, input_values, module_filepath, inject_parameters))

"""
Golden output = tanpa fault injection

faulty_a = a + perturbation
a*b + (perturbation*b) = faulty_res
"""
