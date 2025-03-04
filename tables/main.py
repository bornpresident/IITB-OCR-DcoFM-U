from PIL import Image
import pytesseract
from .physical_tsr import get_cols_from_tatr, get_cells_from_rows_cols, get_rows_from_tatr
from .logical_tsr import get_logical_structure, align_otsl_from_rows_cols, convert_to_html, export_to_onnx, get_tensor_output
from bs4 import BeautifulSoup
import pathlib
import torch
import cv2
import os

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

def get_cell_ocr(img, bbox, lang):
    cell_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cell_pil_img = Image.fromarray(cell_img)
    ocr_result = pytesseract.image_to_string(cell_pil_img, config='--psm 6', lang = lang)
    ocr_result = ocr_result.replace("\n", " ")
    ocr_result = ocr_result[:-1]
    return ocr_result

def order_rows_cols(rows, cols):
    # Order rows from top to bottom based on y1 (second value in the bounding box)
    rows = sorted(rows, key=lambda x: x[1])
    # Order columns from left to right based on x1 (first value in the bounding box)
    cols = sorted(cols, key=lambda x: x[0])
    return rows, cols

def perform_tsr(img_file, x1, y1, struct_only, lang, export_tensor=False, export_onnx=False):
    rows = get_rows_from_tatr(img_file)
    cols = get_cols_from_tatr(img_file)
    print('Physical TSR')
    print(str(len(rows)) + ' rows detected')
    print(str(len(cols)) + ' cols detected')
    rows, cols = order_rows_cols(rows, cols)
    ## Extracting Grid Cells
    cells = get_cells_from_rows_cols(rows, cols)
    ## Corner case if no cells detected
    if len(cells) == 0:
        table_img = cv2.imread(img_file)
        h, w, c = table_img.shape
        bbox = [0, 0, w, h]
        text = get_cell_ocr(table_img, bbox, lang)
        html_string = '<html><table><tr><td>' + text + '</td></tr></table></html>'
        return html_string
    
    print('Logical TSR')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get both OTSL string and raw prediction
    if export_tensor or export_onnx:
        otsl_string, tensor_output = get_logical_structure(img_file, device)
    else:
        otsl_string, _ = get_logical_structure(img_file, device)
    
    # Export the model to ONNX if requested
    if export_onnx:
        output_dir = os.path.dirname(img_file)
        onnx_path = os.path.join(output_dir, "sprint_model.onnx")
        export_to_onnx(onnx_path)
        print(f"ONNX model exported to: {onnx_path}")
    
    corrected_otsl = align_otsl_from_rows_cols(otsl_string, len(rows), len(cols))
    # Correction
    corrected_otsl = corrected_otsl.replace("E", "C")
    corrected_otsl = corrected_otsl.replace("F", "C")
    print('OTSL => ' + otsl_string)
    print("Corrected OTSL => " + corrected_otsl)
    html_string, struc_cells = convert_to_html(corrected_otsl, len(rows), len(cols), cells)
    # Parse the HTML
    soup = BeautifulSoup('<html>' + html_string + '</html>', 'html.parser')

    # Do this if struct_only flag is FALSE
    if struct_only == False:
        cropped_img = cv2.imread(img_file)
        for bbox in soup.find_all('td'):
            # Replace the content inside the div with its 'bbox' attribute value
            ocr_bbox = bbox['bbox'].split(' ')
            ocr_bbox = list(map(int, ocr_bbox))
            bbox.string = get_cell_ocr(cropped_img, ocr_bbox, lang)
            # Correct wrt table coordinates
            ocr_bbox[0] += x1
            ocr_bbox[1] += y1
            ocr_bbox[2] += x1
            ocr_bbox[3] += y1
            bbox['bbox'] = f'{ocr_bbox[0]} {ocr_bbox[1]} {ocr_bbox[2]} {ocr_bbox[3]}'
    
    # Return tensor output if requested
    if export_tensor:
        return soup, struc_cells, tensor_output
    
    return soup, struc_cells

def get_table_hocr(finalimgtoocr, outputDirectory, page, bbox, tab_cnt, lang, export_tensor=False, export_onnx=False):
    image = cv2.imread(finalimgtoocr)
    cropped_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    image_file_name = '/Cropped_Images/table_' + str(page) + '_' + str(tab_cnt) + '.jpg'
    cv2.imwrite(outputDirectory + image_file_name, cropped_image)
    
    if export_tensor or export_onnx:
        soup, cells, tensor_output = perform_tsr(outputDirectory + image_file_name, 0, 0, False, lang, export_tensor, export_onnx)
        if export_tensor:
            # Save tensor output
            tensor_output_path = f"{outputDirectory}/tensor_output_table_{page}_{tab_cnt}.pt"
            torch.save(tensor_output, tensor_output_path)
            print(f"Tensor output saved to: {tensor_output_path}")
        return str(soup)[6:-7] + '\n', tensor_output if export_tensor else None
    else:
        soup, cells = perform_tsr(outputDirectory + image_file_name, 0, 0, False, lang)
        return str(soup)[6:-7] + '\n'

# Function to directly export the model
def export_sprint_model_to_onnx(output_path="sprint_model.onnx"):
    """
    Export the SPRINT model to ONNX format
    
    Args:
        output_path: Path to save the ONNX model
    
    Returns:
        Path to the exported ONNX model
    """
    return export_to_onnx(output_path)

if __name__=="__main__":
    print()
    # Try TD Call
    # tabs = perform_td('samples/page.png')
    # print(tabs)

    # Try Apps call
    # hocr = get_full_page_hocr('samples/page.png', 'eng')
    # print(hocr)

    # Try TSR Call
    # img_path = "samples/cropped_img_2.jpg"
    # hocr_string = perform_tsr(img_path, 0, 0, struct_only = True)
    # print(hocr_string)
    
    # Export model to ONNX
    # export_sprint_model_to_onnx("tables/model/sprint.onnx")
