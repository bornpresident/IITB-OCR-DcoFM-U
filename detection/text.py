import cv2
import pytesseract

def get_text_hocr(image_path, bbox, lang):
    image = cv2.imread(image_path)
    cropped_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    cv2.imwrite('temp.jpg', cropped_image)
    extracted_text = pytesseract.image_to_string('temp.jpg', lang=lang, config='--psm 6')
    hocr = f'<p bbox=\"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\">{extracted_text}</p>\n'
    return hocr