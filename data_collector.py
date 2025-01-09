import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from executor import run_module

def generate_indices_list(size):
    template = [0, 0, 0, 0]
    res = []
    for i in range(size):
        for j in range(size):
            template[2] = i
            template[3] = j
            res.append(template.copy())
    return res

def generate_data(model, size, fault, input_A, input_B, target_indices, faulty_tensor_name, faulty_bit_position):
    input_values = {
        'input_A': input_A,
        'input_B': input_B
    }
    inject_parameters = {
        "inject_type": fault,
        "faulty_tensor_name": faulty_tensor_name,
        "faulty_bit_position": faulty_bit_position,
        "faulty_output_tensor": "output_Y",
        "faulty_operation_name": model[0],
        "target_indices": target_indices
    }
    _, weight_dict = run_module(None, input_values, model[1].format(size), inject_parameters)
    new_row = {
        'op_type': model[0],
        'fault_model': fault,
        'size': size,
        'input_a': str(input_A.tolist()),
        'input_b': str(input_B.tolist()),
        'target_indices': str(target_indices),
        'faulty_tensor': faulty_tensor_name,
        'faulty_bit_position': inject_parameters["faulty_bit_position"],
        'golden_output': str(weight_dict['output_original'].tolist()),
        'faulty_output': str(weight_dict['output_Y'].tolist())
    }
    return new_row

def find_same_faulty(rows):
    grouped = rows.groupby(['input_a', 'input_b', 'op_type', 'fault_model', 'size', 'faulty_output'])
    result = grouped.filter(lambda x: len(x) > 1)
    return result

if __name__ == '__main__':
    output_path_matmul = sys.argv[1]
    output_path_conv = sys.argv[2]
    assert(output_path_matmul is not None and output_path_conv is not None)
    if os.path.exists(output_path_matmul):
        os.remove(output_path_matmul)
    if os.path.exists(output_path_conv):
        os.remove(output_path_conv)

    df = pd.DataFrame(columns=['op_type', 'fault_model', 'size', 'input_a', 'input_b', 'target_indices', 'faulty_tensor', 'faulty_bit_position', 'golden_output', 'faulty_output'])

    matmul_model_path_template = 'models/matmul_integer_{}.onnx'
    conv_integer_path_template = 'models/conv_integer_{}.onnx'

    input = ["input_A", "input_B"]
    sizes = [2, 4, 8, 16, 32, 64]
    # fault_model = ["RANDOM", "RANDOM BITFLIP", "INPUT", "INPUT16", "WEIGHT", "WEIGHT16"]
    fault_model = ["INPUT", "INPUT16", "WEIGHT", "WEIGHT16"]
    models = [('MatMulInteger', matmul_model_path_template), ('ConvInteger', conv_integer_path_template)]

    curr_rows = 0
    total_rows = len(models) * len(sizes) * len(fault_model) * len(input) * 7
    temp = 0
    for size in sizes:
        temp += size * size
    total_rows *= temp

    with tqdm(total=total_rows, desc="Total Progress") as pbar:
        for model in models:
            for size in sizes:
                input_A = None
                input_B = None
                if model[0] == 'MatMulInteger':
                    input_A = np.random.randint(-128, 127, (1, 1, size, size)).astype(np.int8)
                    input_B = np.random.randint(-128, 127, (1, 1, size, size)).astype(np.int8)
                elif model[0] == 'ConvInteger':
                    input_A = np.random.randint(0, 225, (1, 1, size, size)).astype(np.uint8)
                    input_B = np.random.randint(0, 225, (1, 1, size, size)).astype(np.uint8)
                else:
                    raise ValueError("Invalid model type")
                assert(input_A is not None and input_B is not None)
                faulty_indices = generate_indices_list(size)
                for fault in fault_model:
                    for tensor_name in input:
                        for idx in faulty_indices:
                            for bit_pos in range(7):
                                new_row = generate_data(model, size, fault, input_A, input_B, idx, tensor_name, bit_pos)
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                                # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                                # if len(df) >= 10:
                                #     print(f"Processed {len(df)} rows")
                                #     df.to_csv(output_path, mode='a', header=False, index=False)
                                #     df = df.iloc[0:0]
                                #     print(f"DF length: {len(df)}")
                                curr_rows += 1
                                # print(f"Progress: {curr_rows/total_rows * 100}%\t{curr_rows}/{total_rows} rows")
                                pbar.update(1)
                    df = find_same_faulty(df)
                    if model[0] == 'MatMulInteger':
                        df.to_csv(output_path_matmul, mode='a', header=False, index=False)
                    elif model[0] == 'ConvInteger':
                        df.to_csv(output_path_conv, mode='a', header=False, index=False)
                    df = df.iloc[0:0]

    # df.to_csv(output_path, mode='a', header=False, index=False)
