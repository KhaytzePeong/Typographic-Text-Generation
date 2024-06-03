import csv
import glob
import os
import random
import cv2
import numpy as np
import pandas as pd
from unidecode import unidecode
import json
from PIL import Image, ImageDraw, ImageFont
from annotator.canny import CannyDetector

apply_canny = CannyDetector()

def remove_non_ascii(text):
    return unidecode(text)

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def render_font_in_bbox(word, ttf, x, y, w, h, margin=0):
    fs = 100

    # draw word
    position = (x+(w//2), y+(h//2)+1)
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(ttf, fs)
    left, top, right, bottom = draw.textbbox(position, word, font=font, anchor='mm')

    # margin = 2
    while left < (x+margin) or right > (x + w-margin) or top < (y+margin) or bottom >= (y+h-margin):
        fs -= 1
        font = ImageFont.truetype(ttf, fs)
        left, top, right, bottom = draw.textbbox(position, word, font=font, anchor='mm')
    # font = ImageFont.truetype(ttf, fs)
    draw.text(position, word, font=font, fill="black", anchor='mm', align='center')

    return image

def preprocess_font(ori_image_path, mask_path, word, font):
    img = Image.open(ori_image_path)
    img = np.array(img)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(mask_contours[0])

    font_img = render_font_in_bbox(word, font, x, y, w, h, 1)
    np_font_img = np.array(font_img)
    font_canny = apply_canny(np_font_img, 100, 200)

    return img, font_canny, mask
    
def get_bbox(image, isbinary=False):
    '''
    get bbox coords of grayscale image
    '''

    if not isbinary:
        # convert to binary image
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        binary_image = image
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the overall bounding box
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    # Iterate through contours to get individual bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Update overall bounding box
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    
    x = min_x
    y = min_y
    w = max_x - min_x
    h = max_y - min_y

    return x, y, w, h

def render_mask(x, y, w, h, image_size=(512, 512)):
    '''
    render new mask given bbox coords
    '''
    
    # Create a blank image with a white background
    img = Image.new('L', image_size, color=0)
    draw = ImageDraw.Draw(img)

    # Draw a filled rectangle (black) based on the provided x, y, w, h
    draw.rectangle([x, y, x + w, y + h], fill=255)

    return img