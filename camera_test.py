import cv2
import detection_utility
import recognition_utility

DETECTION_MODEL_RES = (detection_utility.IMAGE_WIDTH,
                       detection_utility.IMAGE_HEIGHT)
OUTPUT_RES = (1280, 720)

if (__name__ == '__main__'):
    detection_model = detection_utility.load_model(
        "exported_models/ssd_mobilenet_v2_phone_numbers/saved_model")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        mobilenet_frame = cv2.resize(frame, DETECTION_MODEL_RES)
        output_dict = detection_utility.get_prediction_dict(detection_model,
                                                            mobilenet_frame)
        start_point, end_point = detection_utility. \
            get_best_box(output_dict['detection_boxes'],
                         output_dict['detection_scores'])
        if start_point != (-1, -1):  # when there is no objects on frame
            mobilenet_frame = recognition_utility. \
                draw_rect_on_frame(mobilenet_frame,
                                   '777',
                                   start_point,
                                   end_point)
            frame = cv2.resize(mobilenet_frame, OUTPUT_RES)

        frame = cv2.resize(frame, OUTPUT_RES)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
