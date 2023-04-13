from tensorflow.keras.models import load_model as load

import cv2
import numpy as np

MIN_AREA = 400  # like 20x20 pixels for one number
MAX_AREA = 10000  # like 100x100 pixel for one number
OUTPUT_SIZE = (28, 28)


def load_model(path: str):
    model = load(path)
    return model


def predict(recognition_model, filtered_boxes):
    reshaped = list()
    for box in filtered_boxes:
        reshaped.append(box[1])
    x_test = np.array(reshaped)
    prediction = recognition_model.predict(x_test)
    phone_number = [np.argmax(number) for number in prediction]
    return ''.join(map(str, phone_number))


def crop_frame(frame, start_point: tuple, end_point: tuple):
    x1, y1 = start_point
    x2, y2 = end_point
    return frame[y1:y2, x1:x2]


def scale(image, clip_hist_percent=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = []
    accumulator.append(float(hist[0]))

    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    minimum_gray = 0

    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    scaled = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return scaled


def get_contours(image):
    if 0 not in image.shape:
        autoscaled = scale(image)
        gray = cv2.cvtColor(autoscaled, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        image_erode = cv2.erode(thresh,
                                np.ones((3, 3), np.uint8),
                                iterations=1)
        contours, hierarchy = cv2.findContours(image_erode,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        filtered_boxes = list()
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            area = w*h
            if (hierarchy[0][idx][3] == 0 and
                    area > MIN_AREA and
                    area < MAX_AREA):
                filtered_boxes.append((x,
                                       cv2.resize(gray[y:y+h, x:x+w],
                                                  OUTPUT_SIZE)))
        filtered_boxes.sort(key=lambda x: x[0])  # sort from left to right
        return filtered_boxes
    return [-1]


def draw_rect_on_frame(frame, phone_number: str,
                       start_point: tuple,
                       end_point: tuple):
    start_x, _ = start_point
    _, end_y = end_point
    frame = cv2.rectangle(frame, start_point, end_point,
                          color=(0, 0, 0), thickness=1)
    frame = cv2.putText(frame, phone_number, (start_x + 5, end_y + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    return frame
