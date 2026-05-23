import os
import cv2
import sys
import argparse

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
base_path = os.path.join(realpath[0] + _sep, *realpath[1:realpath.index('rknn_model_zoo') + 1])
sys.path.append(base_path)

from py_utils.coco_utils import COCO_test_helper
import numpy as np

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

IMG_SIZE = (640, 640)

# ============================================================
# 自定义类别 (工具检测 - 25 类)
# ============================================================
CLASSES = (
    "活口扳手", "示波器", "油壶", "铲子", "塞尺",
    "手摇把", "加封钳子", "医用钳", "转辙机钥匙", "插片",
    "扭力扳手", "扭力扳手盒", "测距仪", "手电筒", "两用扳手",
    "锤子", "尖嘴钳", "套筒", "卷尺", "老虎钳",
    "螺丝刀", "斜口钳", "禁动牌", "胶带", "笔",
)


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

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

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    keep = np.array(keep)
    return keep


def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c_ = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c_[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cls_id = int(cl)
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        print("%s @ (%d %d %d %d) %.3f" % (cls_name, top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(cls_name, score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} 不是 rknn/pytorch/onnx 模型".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform


def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO11 工具检测 RKNN 推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径 (.pt / .onnx / .rknn)')
    parser.add_argument('--target', type=str, default='rk3588', help='目标 RKNPU 平台')
    parser.add_argument('--device_id', type=str, default=None, help='设备 ID')
    parser.add_argument('--img_show', action='store_true', default=False, help='显示检测结果')
    parser.add_argument('--img_save', action='store_true', default=False, help='保存检测结果')
    parser.add_argument('--img_folder', type=str, default='./images', help='图片文件夹路径')

    args = parser.parse_args()

    model, platform = setup_model(args)

    if os.path.isdir(args.img_folder):
        file_list = sorted(os.listdir(args.img_folder))
        img_list = [path for path in file_list if img_check(path)]
    elif os.path.isfile(args.img_folder):
        img_list = [os.path.basename(args.img_folder)]
        args.img_folder = os.path.dirname(args.img_folder)
    else:
        print("错误: 找不到图片路径: {}".format(args.img_folder))
        exit(1)

    co_helper = COCO_test_helper(enable_letter_box=True)

    for i in range(len(img_list)):
        print('检测中 {}/{}'.format(i + 1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} 找不到")
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        pad_color = (0, 0, 0)
        img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                                   pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = post_process(outputs)

        if args.img_show or args.img_save:
            print('\n\n图片: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            if args.img_save:
                result_dir = './result'
                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)
                result_path = os.path.join(result_dir, img_name)
                cv2.imwrite(result_path, img_p)
                print('结果保存至 {}'.format(result_path))

            if args.img_show:
                cv2.imshow("检测结果", img_p)
                cv2.waitKeyEx(0)

    model.release()
    print('\n✅ 推理完成')