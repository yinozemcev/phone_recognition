import cv2
import detection_utility
import recognition_utility

DETECTION_MODEL_RES = (detection_utility.IMAGE_WIDTH,
                       detection_utility.IMAGE_HEIGHT)
OUTPUT_RES = (1280, 720)
NUMBERS_IN_PHONE = 11

if (__name__ == '__main__'):
    detection_model = detection_utility.load_model(
        "exported_models/ssd_mobilenet_v2_phone_numbers/saved_model")
    recognition_model = recognition_utility.load_model('ConvRecognition.keras')

    cap = cv2.VideoCapture(0)

    while True:
        stop = False
        ret, frame = cap.read()
        if ret:
            mobilenet_frame = cv2.resize(frame, DETECTION_MODEL_RES)
            output_dict = detection_utility. \
                get_prediction_dict(detection_model,
                                    mobilenet_frame)
            start_point, end_point = detection_utility. \
                get_best_box(output_dict['detection_boxes'],
                             output_dict['detection_scores'])
            if start_point != (-1, -1):  # when there is no objects on frame
                cropped_frame = recognition_utility.crop_frame(mobilenet_frame,
                                                               start_point,
                                                               end_point)
                countoured_numbers = recognition_utility. \
                    get_contours(cropped_frame)

                if len(countoured_numbers) == NUMBERS_IN_PHONE:
                    recognized_phone = recognition_utility. \
                        predict(recognition_model,
                                countoured_numbers)
                    mobilenet_frame = recognition_utility. \
                        draw_rect_on_frame(mobilenet_frame,
                                           recognized_phone,
                                           start_point,
                                           end_point)

                    frame = cv2.resize(mobilenet_frame, OUTPUT_RES)
                    stop = True

            frame = cv2.resize(frame, OUTPUT_RES)
        cv2.imshow('frame', frame)

        if stop:
            cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
