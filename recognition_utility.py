import cv2


def load_model(path: str):
    pass


def draw_rect_on_frame(frame, phone_number: str,
                       start_point: tuple,
                       end_point: tuple):
    start_x, _ = start_point
    _, end_y = end_point
    frame = cv2.rectangle(frame, start_point, end_point,
                          color=(0, 0, 0), thickness=1)
    frame = cv2.putText(frame, phone_number, (start_x + 5, end_y + 5),
                        font=cv2.FONT_HERSHEY_DUPLEX,
                        font_scale=1.0,
                        color=(255, 255, 255),
                        thickness=1)

    return frame
