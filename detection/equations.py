from PIL import Image
from pix2tex.cli import LatexOCR
import cv2

LatexModelEquation = LatexOCR()

def get_equation_recognition(image_path):
    img_pil = Image.open(image_path)
    return LatexModelEquation(img_pil)

def get_equation_hocr(image_path, outputDirectory, pagenumber, eqn_bbox, count):
    image = cv2.imread(image_path)
    fig = eqn_bbox
    cropped_image = image[fig[1]: fig[3], fig[0]: fig[2]]
    image_file_name = '/Cropped_Images/equation_' + str(pagenumber) + '_' + str(count) + '.jpg'
    cv2.imwrite(outputDirectory + image_file_name, cropped_image)
    equation_recog = get_equation_recognition(outputDirectory + image_file_name)
    eqn_hocr = f'<span bbox=\"{fig[0]} {fig[1]} {fig[2]} {fig[3]}\">{equation_recog}</span>\n'
    return eqn_hocr

if __name__ == "__main__":
    print('YES')