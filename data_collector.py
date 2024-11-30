from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from executor import run_module

def generate_data(model, size, fault):
    input_A = None
    input_B = None
    if model[0] == 'MatMulInteger':
        input_A = np.random.randint(-128, 127, (1, 1, size, size)).astype(np.int8)
        input_B = np.random.randint(-128, 127, (1, 1, size, size)).astype(np.int8)
    elif model[0] == 'ConvInteger':
        input_A = np.random.randint(0, 225, (1, 1, size, size)).astype(np.uint8)
        input_B = np.random.randint(0, 225, (1, 1, size, size)).astype(np.uint8)
    else:
        raise ValueError('Invalid model type')
    assert(input_A is not None and input_B is not None)
    input_values = {
        'input_A': input_A,
        'input_B': input_B
    }
    inject_parameters = {
        "inject_type": fault,
        "faulty_tensor_name": "input_A",
        "faulty_bit_position": np.random.randint(0, 7),
        "faulty_output_tensor": "output_Y",
        "faulty_operation_name": model[0]
    }
    _, weight_dict = run_module(None, input_values, model[1].format(size), inject_parameters)
    new_row = {
        'op_type': model[0],
        'fault_model': fault,
        'size': size,
        'input_a': input_A.tolist(),
        'input_b': input_B.tolist(),
        'target_indices': weight_dict['target_indices'],
        'faulty_bit_position': inject_parameters["faulty_bit_position"],
        'golden_output': weight_dict['output_original'].tolist(),
        'faulty_output': weight_dict['output_Y'].tolist()
    }
    return new_row

if __name__ == '__main__':
    df = pd.DataFrame(columns=['op_type', 'fault_model', 'size', 'input_a', 'input_b', 'target_indices', 'faulty_bit_position', 'golden_output', 'faulty_output'])

    matmul_model_path_template = 'models/matmul_integer_{}.onnx'
    conv_integer_path_template = 'models/conv_integer_{}.onnx'

    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    fault_model = ["RANDOM", "RANDOM BITFLIP", "INPUT", "INPUT16", "WEIGHT", "WEIGHT16"]
    models = [('MatMulInteger', matmul_model_path_template), ('ConvInteger', conv_integer_path_template)]

    with ThreadPoolExecutor() as executor:
        futures = []
        for model in models:
            for size in sizes:
                for fault in fault_model:
                    for i in range(50):
                        futures.append(executor.submit(generate_data, model, size, fault))

        for future in futures:
            new_row = future.result()
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv('data.csv', index=False)
