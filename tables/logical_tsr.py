import torch
from PIL import Image
from torchvision import transforms
import torch.onnx
import os

def convert_otsl_list(otsl_list):
    final_seq = []
    for e in otsl_list:
        if e == 'fcel' or e == 'ecel':
            final_seq.append('C')
        elif e == 'lcel':
            final_seq.append('L')
        elif e == 'ucel':
            final_seq.append('U')
        elif e == 'xcel':
            final_seq.append('X')
        elif e == 'nl':
            final_seq.append('N')
        else:
            final_seq.append('C')
    return final_seq

def count_contiguous_occurrences(s, target_char):
    count = 0
    for char in s:
        if char == target_char:
            count += 1
        else:
            break
    return count

def get_cell_spans(otsl_matrix, i, j):
    entry = otsl_matrix[i][j]
    if entry != 'C':
        return 0, 0
    else:
        row_seq = ''.join(otsl_matrix[i])[j + 1:]
        col_seq = ''.join(row[j] for row in otsl_matrix)[i + 1:]
        rs = count_contiguous_occurrences(row_seq, 'L')
        cs = count_contiguous_occurrences(col_seq, 'U')
        return rs, cs

def get_conv_html_from_otsl_with_cells(otsl_matrix, R, C, cells):
    html_string = '<table><tbody>'
    struc_cells = []
    # Generate string
    for i in range(R):
        html_string += '<tr>'
        for j in range(C + 1):
            e = otsl_matrix[i][j]
            if e == 'C':
                td_cell = cells[i + 1][j]
                rs, cs = get_cell_spans(otsl_matrix, i, j)
                if rs and cs:
                    # There is rowspan and colspan
                    extension = cells[i + 1 + cs][j + rs]
                    td_cell = [td_cell[0], td_cell[1], extension[2], extension[3]]
                    html_string += f'<td rowpsan="{rs + 1}" colspan="{cs + 1}" bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                elif rs and not cs:
                    # There is only row span
                    extension = cells[i + 1][j + rs]
                    td_cell = [td_cell[0], td_cell[1], extension[2], td_cell[3]]
                    html_string += f'<td colspan="{rs + 1}" bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                elif not rs and cs:
                    # There is only col span
                    extension = cells[i + 1 + cs][j]
                    td_cell = [td_cell[0], td_cell[1], td_cell[2], extension[3]]
                    html_string += f'<td rowspan="{cs + 1}" bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                else:
                    # Normal cell
                    html_string += f'<td bbox="{td_cell[0]} {td_cell[1]} {td_cell[2]} {td_cell[3]}"></td>'
                struc_cells.append(td_cell)
            elif e == 'N':
                # New row will start
                html_string += '</tr>'
            else:
                continue
    html_string += '</tbody></table>'
    return html_string, struc_cells

def align_otsl_from_rows_cols(otsl_string, rows, cols):
    N = len(otsl_string)
    C = cols
    R = rows
    if N != (C + 1) * R:
        # Needs correction
        actual_N = (C + 1) * R
        if N > actual_N:
            otsl_string = otsl_string[:actual_N]
            otsl_string = otsl_string[:-1] + 'N'
        else:
            diff = actual_N - N
            suffix = 'C' * (diff - 1) + 'N'
            otsl_string = otsl_string + suffix

    # Make sure Ns are at correct position !!
    # Remove if N is misplaced
    otsl_string_list = []
    for i in range(len(list(otsl_string))):
        char = otsl_string[i]
        if i > 0 and (i + 1) % (C + 1) == 0:
            char = 'N'
        else:
            if otsl_string[i] == 'N':
                char = 'C'
        otsl_string_list.append(char)
    final_otsl_string = ''.join(otsl_string_list)
    return final_otsl_string

def convert_to_html(otsl_string, R, C, cells):
    N = len(otsl_string)
    if N != (C + 1) * R:
        # Needs correction
        actual_N = (C + 1) * R
        if N > actual_N:
            otsl_string = otsl_string[:actual_N]
            otsl_string = otsl_string[:-1] + 'N'
        else:
            diff = actual_N - N
            suffix = 'C' * (diff - 1) + 'N'
            otsl_string = otsl_string + suffix

    # Init OTSL matrix
    otsl_matrix = [[otsl_string[i * (C + 1) + j] for j in range(C + 1)] for i in range(R)]

    # Handle for 'U' in first row, replace by 'C'
    for i in range(len(otsl_matrix[0])):
        if otsl_matrix[0][i] == 'U':
            otsl_matrix[0][i] = 'C'

    # Handle for L in first column, replace by 'C'
    for i in range(R):
        if otsl_matrix[i][0] == 'L':
            otsl_matrix[i][0] = 'C'

    # Return converted string
    return get_conv_html_from_otsl_with_cells(otsl_matrix, R, C, cells)

def get_logical_structure(img_file, device):
    # Load image
    img = Image.open(img_file)
    img = test_transforms(img)
    input_img = torch.stack([img])

    # if not torch.device("cpu"):
    print('SPRINT inference on ' + str(device))
    input_img = input_img.to(device)

    # Infer
    pred = model(input_img, None, return_preds=True)
    otsl = pred['preds'][0][0]
    
    # Return both the OTSL string and the raw prediction tensor
    return otsl, pred

# Function to export model to ONNX format
def export_to_onnx(output_path="sprint_model.onnx", batch_size=1):
    """
    Export the loaded SPRINT model to ONNX format
    Args:
        output_path: path to save the ONNX model
        batch_size: batch size for the model input
    """
    print(f"Exporting model to ONNX format: {output_path}")
    
    # Create a dummy input tensor with the appropriate shape
    dummy_input = torch.randn(batch_size, 3, 128, 128, device=device)
    
    # Additional inputs required by the model (if any)
    dummy_labels = None
    
    # Export the model
    torch.onnx.export(
        model,                                    # model being run
        (dummy_input, dummy_labels),              # model input (or a tuple for multiple inputs)
        output_path,                              # where to save the model
        export_params=True,                       # store the trained parameter weights inside the model file
        opset_version=12,                         # the ONNX version to export the model to
        do_constant_folding=True,                 # whether to execute constant folding for optimization
        input_names=['input', 'labels'],          # the model's input names
        output_names=['output'],                  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},           # variable length axes
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    
    print(f"Model exported successfully to {output_path}")
    return output_path

# Function to get tensor output directly
def get_tensor_output(img_file, device):
    """
    Get the raw tensor output from the model
    Args:
        img_file: path to the image file
        device: device to run inference on
    Returns:
        raw tensor output from the model
    """
    # Load image
    img = Image.open(img_file)
    img = test_transforms(img)
    input_img = torch.stack([img])
    input_img = input_img.to(device)
    
    # Infer and return raw output
    with torch.no_grad():
        output = model(input_img, None, return_preds=True)
    
    return output

# Config
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
model_path = 'tables/model/sprint.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# If this file is run directly, export the model to ONNX
if __name__ == "__main__":
    print("Exporting SPRINT model to ONNX format...")
    export_path = export_to_onnx()
    print(f"ONNX model exported to: {export_path}")
