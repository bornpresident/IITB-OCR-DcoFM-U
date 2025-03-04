import streamlit as st
from perform_ocr import pdf_to_txt
import zipfile, os
from config import input_dir
import pytesseract


def save_uploaded_file(uploadedfile):
    with open(os.path.join(input_dir, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

st.title('Custom HOCR for DocFM')
input_file = st.file_uploader('Choose your .pdf file', type="pdf")
outputsetname = st.text_input(label= "Enter output set name here", value="")
language = st.text_input(label= "Enter language here", value="eng")
langs = pytesseract.get_languages()
avail_langs = 'Available languages are : ' + str(langs)
st.text(avail_langs)

# Add options for tensor export and ONNX export
export_tensor = st.checkbox("Export model tensors", value=False, help="Export raw tensor outputs for tables")
export_onnx = st.checkbox("Export model to ONNX", value=False, help="Export the SPRINT model to ONNX format")

if len(outputsetname) and input_file is not None:
    go = st.button("Get OCR")
    if go:
        save_uploaded_file(input_file)
        with st.spinner('Loading...'):
            outputDirectory = pdf_to_txt(input_dir + input_file.name, outputsetname, language, export_tensor, export_onnx)

        zipfile_name = outputDirectory + '.zip'
        zf = zipfile.ZipFile(zipfile_name, "w")
        for dirname, subdirs, files in os.walk(outputDirectory):
            zf.write(dirname)
            for filename in files:
                zf.write(os.path.join(dirname, filename))
        zf.close()

        with open(zipfile_name, "rb") as fp:
            btn = st.download_button(
                label = "Download ZIP",
                data = fp,
                file_name = f'{outputsetname}.zip',
                mime = "application/zip"
            )
