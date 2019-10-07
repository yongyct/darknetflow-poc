import os
import cv2
import numpy as np

from darknetflow_poc.utils.constants import IMAGE_EXTENSIONS


def get_input_images(conf):
    """
    Get valid input files in provided data directory within json config
    :param conf: user provided json config
    :return: list of valid image files that can be used
    """
    data_dir = conf.IN_DATA_DIR
    return [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)
            if file_name.split('.')[-1].lower() in IMAGE_EXTENSIONS]


def get_preprocessed_image(image, dim):
    """
    Pre-process input image and returns it as a numpy ndarray
    :param image: input image to be preprocessed
    :return: numpy representation of image
    """
    # img_norm = (255 - cv2.resize(cv2.imread(image, 0), (40, 30), interpolation=cv2.INTER_AREA)) / 255
    img_norm = cv2.resize(cv2.imread(image), (dim[0], dim[1]), interpolation=cv2.INTER_AREA) / 255
    return np.expand_dims(a=img_norm, axis=0)


def get_all_class_labels(conf):
    """
    Reads class labels from supplied text file in json config
    :param conf: user provided json config
    :return: list of class labels to be detected
    """
    with open(conf.LABELS_PATH) as labels_file:
        class_labels = labels_file.readlines()
    return [class_label.strip for class_label in class_labels]


def draw_bounding_boxes(image, boxes, scores, classes, all_classes):
    """
    Draw bounding boxes around predictions
    :param image: ndarray, original image
    :param boxes: ndarray, boxes of objects
    :param scores: ndarray, scores of objects
    :param classes: ndarray, classes of objects
    :param all_classes: list, all class labels
    :return: None
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            '{} {:.2f}'.format(all_classes[int(cl)], score),
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )


def yolo_out(outs, shape, conf):
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []

    for out, mask in zip(outs, masks):
        b, c, s = _process_output_features(out, anchors, mask)
        b, c, s = _filter_boxes(b, c, s, conf.OBJECT_THRESHOLD)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # Scale boxes back to original image shape.
    width, height = shape[1], shape[0]
    image_dims = [width, height, width, height]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = _nms_boxes(b, s, conf.NMS_THRESHOLD)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _process_output_features(out, anchors, mask):
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = _sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_wh = box_wh * anchors_tensor

    box_confidence = _sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _sigmoid(out[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs


def _filter_boxes(boxes, box_confidences, box_class_probs, threshold):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= threshold)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores


def _nms_boxes(boxes, scores, threshold):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep
