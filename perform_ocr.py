from PIL import Image
import os
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
import time
import sys
from config import output_dir, config_dir
from detection import get_page_layout, get_equation_hocr, get_figure_hocr, get_text_hocr
from tables import get_table_hocr, export_sprint_model_to_onnx
import torch

def parse_boolean(b):
    return b == "True"

# For simpler filename generation
def simple_counter_generator(prefix="", suffix=""):
    i = 400
    while True:
        i += 1
        yield 'p'

def pdf_to_txt(orig_pdf_path, project_folder_name, lang, export_tensor=False, export_onnx=False):
    outputDirIn = output_dir
    outputDirectory = outputDirIn + project_folder_name
    print('Output directory is ', outputDirectory)
    # create images,text folder
    print('cwd is ', os.getcwd())
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)

    if not os.path.exists(outputDirectory + "/Images"):
        os.mkdir(outputDirectory + "/Images")

    imagesFolder = outputDirectory + "/Images"
    imageConvertOption = 'True'

    print("converting pdf to images")
    jpegopt = {
        "quality": 100,
        "progressive": True,
        "optimize": False
    }

    output_file = simple_counter_generator("page", ".jpg")
    print('orig pdf oath is', orig_pdf_path)
    print('cwd is', os.getcwd())
    print("orig_pdf_path is", orig_pdf_path)
    if (parse_boolean(imageConvertOption)):
        convert_from_path(orig_pdf_path, output_folder=imagesFolder, dpi=300, fmt='jpeg', jpegopt=jpegopt,
                          output_file=output_file)

    print("images created.")
    print("Now we will OCR")
    os.environ['IMAGESFOLDER'] = imagesFolder
    os.environ['OUTPUTDIRECTORY'] = outputDirectory
    print("Selected language model " + lang)
    if not os.path.exists(outputDirectory + "/Cropped_Images"):
        os.mkdir(outputDirectory + "/Cropped_Images")
    if not os.path.exists(outputDirectory + "/Inds"):
        os.mkdir(outputDirectory + "/Inds")
        os.mknod(outputDirectory + "/Inds/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/Layout_Images"):
        os.mkdir(outputDirectory + "/Layout_Images")
    
    # Create tensor outputs directory if exporting tensors
    if export_tensor and not os.path.exists(outputDirectory + "/Tensor_Outputs"):
        os.mkdir(outputDirectory + "/Tensor_Outputs")
        
    # Export ONNX model if requested
    if export_onnx:
        onnx_path = os.path.join(outputDirectory, "sprint_model.onnx")
        exported_path = export_sprint_model_to_onnx(onnx_path)
        print(f"SPRINT model exported to ONNX: {exported_path}")

    individualOutputDir = outputDirectory + "/Inds"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    startOCR = time.time()

    for imfile in os.listdir(imagesFolder):
        finalimgtoocr = imagesFolder + "/" + imfile
        dash = imfile.index('-')
        dot = imfile.index('.')
        page = int(imfile[dash + 1 : dot])
        layout_annotated_image_path = outputDirectory + "/Layout_Images/" + imfile
        dets = get_page_layout(finalimgtoocr, layout_annotated_image_path, device)
        dets.sort(key = lambda x : x[1][1])
        hocr_elements = ''
        eqn_cnt = 1
        tab_cnt = 1
        # class_names = {0: 'title', 1: 'plain_text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
        for det in dets:
            cls = det[0]
            bbox = det[1]
            if cls == 8:
                # Equations
                eqn_hocr = get_equation_hocr(finalimgtoocr, outputDirectory, page, bbox, eqn_cnt)
                eqn_cnt += 1
                hocr_elements += eqn_hocr
            elif cls == 3:
                # Figures
                fig_hocr = get_figure_hocr(bbox)
                # fig_hocr = get_figure_hocr(finalimgtoocr, outputDirectory, page, bbox, fig_cnt)
                # fig_cnt += 1
                hocr_elements += fig_hocr
            elif cls == 5:
                # Tables
                if export_tensor:
                    tab_hocr, tensor_output = get_table_hocr(finalimgtoocr, outputDirectory, page, bbox, tab_cnt, lang, export_tensor, False)
                    # Save tensor output
                    tensor_output_path = f"{outputDirectory}/Tensor_Outputs/table_{page}_{tab_cnt}.pt"
                    torch.save(tensor_output, tensor_output_path)
                    print(f"Tensor output saved to: {tensor_output_path}")
                else:
                    tab_hocr = get_table_hocr(finalimgtoocr, outputDirectory, page, bbox, tab_cnt, lang, False, False)
                tab_cnt += 1
                hocr_elements += tab_hocr
            else:
                hocr = get_text_hocr(finalimgtoocr, bbox, lang)
                # Can use class_names[cls] for classname instead
                hocr_elements += hocr

        # Now we generate HOCRs using Tesseract
        print('We will OCR the image ' + finalimgtoocr)
        final_hocr = '<html><body>' + hocr_elements + '</body></html>'
        soup = BeautifulSoup(final_hocr, 'html.parser')

        # Write final hocrs
        hocrfile = individualOutputDir + '/' + imfile[:-3] + 'hocr'
        f = open(hocrfile, 'w+')
        f.write(str(soup))

    # Calculate the time elapsed for entire OCR process
    endOCR = time.time()
    ocr_duration = round((endOCR - startOCR), 3)
    print('Done with OCR of ' + str(project_folder_name) + ' of ' + str(len(os.listdir(imagesFolder))) + ' pages in ' + str(ocr_duration) + ' seconds')
    return outputDirectory

# Function Calls
if __name__ == "__main__":
    input_file = sys.argv[1]
    outputsetname = sys.argv[2]
    lang = sys.argv[3]
    export_tensor = False
    export_onnx = False
    
    if len(sys.argv) > 4:
        export_tensor = parse_boolean(sys.argv[4])
    if len(sys.argv) > 5:
        export_onnx = parse_boolean(sys.argv[5])
        
    pdf_to_txt(input_file, outputsetname, lang, export_tensor, export_onnx)
