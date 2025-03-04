from perform_ocr import pdf_to_txt
import sys

# Default parameters
input_file = 'data/input/1.pdf'
outputsetname = '2-nms'
lang = 'eng'
export_tensor = False  # Default: don't export tensor
export_onnx = False    # Default: don't export to ONNX

# Parse command line arguments if provided
if len(sys.argv) > 1:
    input_file = sys.argv[1]
if len(sys.argv) > 2:
    outputsetname = sys.argv[2]
if len(sys.argv) > 3:
    lang = sys.argv[3]
if len(sys.argv) > 4:
    export_tensor = sys.argv[4].lower
