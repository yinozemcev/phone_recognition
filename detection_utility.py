import pathlib
import numpy as np
import tensorflow as tf

from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils.label_map_util \
    import create_category_index_from_labelmap \
    as create_category_index
from object_detection.utils.visualization_utils \
    import visualize_boxes_and_labels_on_image_array \
    as visualize_boxes

PATH_TO_LABELS = 'annotations/object-detection.pbtxt'
TEST_IMAGE_PATHS = sorted(list(pathlib.Path('images/test').glob("*.jp*g")))
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
DETECTION_THRESHOLD = 0.9
BOX_COORDINATE_CORRECTION = 10
AREA_THRESHOLD = 100

# patch tf1 compatibility and gfile
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile


def load_model(path: str):
    model_dir = pathlib.Path(path)
    model = tf.saved_model.load(str(model_dir))
    return model


def get_prediction_dict(model, image: Image) -> dict:
    input_tensor = tf.convert_to_tensor(np.asarray(image))
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = \
        output_dict['detection_classes'].astype(np.int64)

    return output_dict


def display_prediction_to_image(model, image_path: str) -> None:
    image = np.array(Image.open(image_path))
    display(Image.fromarray(get_prediction_to_image(model, image)))


def get_prediction_to_image(model, image: Image) -> Image:
    output = get_prediction_dict(model, image)
    category_index = create_category_index(PATH_TO_LABELS,
                                           use_display_name=True)
    visualize_boxes(image,
                    output['detection_boxes'],
                    output['detection_classes'],
                    output['detection_scores'],
                    category_index,
                    instance_masks=output.get('detection_masks_reframed',
                                              None),
                    use_normalized_coordinates=True,
                    line_thickness=2)
    return image


def get_best_box(detection_boxes: np.array,
                 detection_scores: np.array) -> tuple[tuple]:
    """
    Converts detection box coordinates (ymin, xmin, ymax, xmax) to cv2 format
    for rectangle() function - (start_point, end_point)
    if detection score is more than DETECTION_THRESHOLD (by default 0.7)
    else return (-1, -1), (-1, -1)
    """
    best_score_index = np.argmax(detection_scores)
    if detection_scores[best_score_index] > DETECTION_THRESHOLD:
        y_min, x_min, y_max, x_max = detection_boxes[best_score_index]
        x_min_real = round(x_min*IMAGE_WIDTH) - BOX_COORDINATE_CORRECTION
        y_min_real = round(y_min*IMAGE_HEIGHT) - BOX_COORDINATE_CORRECTION
        x_max_real = round(x_max*IMAGE_WIDTH) + BOX_COORDINATE_CORRECTION
        y_max_real = round(y_max*IMAGE_HEIGHT) + BOX_COORDINATE_CORRECTION
        area = (x_max_real - x_min_real)*(y_max_real - y_min_real)
        if area > AREA_THRESHOLD:
            start_point = (x_min_real, y_min_real)
            end_point = (x_max_real, y_max_real)
            return start_point, end_point
    return (-1, -1), (-1, -1)
