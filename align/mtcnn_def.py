from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
import os
import align.detect_face
from os.path import join as pjoin
from PIL import ImageFont, ImageDraw, Image
from skimage import transform as trans
import cv2

# mark_color = (205, 255, 0)
mark_color = (225, 209, 0)

fontpath = "data_pro/wryh.ttf"  # 32为字体大小
font22 = ImageFont.truetype(fontpath, 22)


# http://www.sfinst.com/?p=1683


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def read_img(person_dir, f):
    img = cv2.imread(pjoin(person_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 判断数组维度
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img


def load_data(data_dir):
    data = {}
    pics_ctr = 0
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
        # 存储每一类人的文件夹内所有图片
        data[guy] = curr_pics
    return data


def embs_toget_names(detect_face_embs_i, known_names_i, known_embs_i):
    L2_dis = np.linalg.norm(detect_face_embs_i - known_embs_i, axis=1)
    is_known = 0
    if min(L2_dis) < 0.55:
        loc_similar_most = np.where(L2_dis == min(L2_dis))
        is_known = 1
        return known_names_i[loc_similar_most][0], is_known
    else:
        return '未知的同学', is_known


def turn_face(b_boxes, points5):
    # bounding_boxes=[[x1,y1,x2,y2,score], [x1,y1,x2,y2,score]]，points_5 = [[x1左眼，x2左眼], [x1右眼，x2右眼],
    # [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    # 初始化存储变量
    b_boxes_new, points5_new = b_boxes, points5

    # 关键点图像坐标转化为直角坐标系坐标，即y*(-1)
    points5[5:] = points5[5:] * (-1)

    # 迭代每一张脸
    for fi in range(len(b_boxes)):

        print('b_boxes[fi], points5 >> fi')
        print(b_boxes[fi])
        for pi in range(10):
            print(points5[pi][fi])
            L = np.asarray([points5[0][fi], points5[5][fi]])
            R = np.asarray([points5[1][fi], points5[6][fi]])
            A_v = R - L
            X_v = np.asarray([1, 0])
            cos_a = np.dot(A_v, X_v.T) / (np.linalg.norm(A_v) * np.linalg.norm(X_v))
            sin_a = np.power(1 - np.power(cos_a, 2), 0.5)

            # box图像坐标系转换为直角坐标系
            b_boxes[fi][1] = b_boxes[fi][1] * (-1)
            b_boxes[fi][3] = b_boxes[fi][3] * (-1)
            P1x = b_boxes[fi][0]
            P1y = b_boxes[fi][1]
            P2x = b_boxes[fi][2]
            P2y = b_boxes[fi][3]
            Cx = b_boxes[fi][0] + (b_boxes[fi][2] - b_boxes[fi][0]) / 2
            Cy = b_boxes[fi][1] + (b_boxes[fi][3] - b_boxes[fi][1]) / 2

            P1_xnew = (P1x - Cx) * cos_a - (P1y - Cy) * sin_a + Cx
            P1_ynew = (P1x - Cx) * sin_a + (P1y - Cy) * cos_a + Cx

            P2_xnew = (P2x - Cx) * cos_a - (P2y - Cy) * sin_a + Cx
            P2_ynew = (P2x - Cx) * sin_a + (P2y - Cy) * cos_a + Cx

            b_boxes_new[fi][0] = P1_xnew
            b_boxes_new[fi][1] = P1_ynew
            b_boxes_new[fi][2] = P2_xnew
            b_boxes_new[fi][3] = P2_ynew

            # 计算原始里面，关键点距离原始左上角的距离
            # b_boes_new进行旋转成正型，
            # xin的dets进行cv2旋转取角度值a时， cos_a = 根号2/2，右眼在上，cos值取小的45度，左眼在上cos值取大的315度

    return b_boxes_new, points5_new


def load_and_align_data(image, image_size, minsize=100):  # 返回彩图
    # face detection parameters
    # 以下两个阈值调整后，歪脸和遮挡会被过滤掉

    # threshold = [0.6, 0.7, 0.7]  # three steps's threshold 三步的阈值
    threshold = [0.85, 0.7, 0.7]  # 第一步pnet阈值升高，候选框会比较少

    # factor = 0.709  # scale factor 比例因子，越小越快
    factor = 0.4  # scale factor 比例因子，图像金字塔每次缩小的保留比例，因此factor值越小越快

    # 读取图片
    img = to_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 ）
    bounding_boxes, points_5 = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # bounding_boxes=[[x1,y1,x2,y2,score], [x1,y1,x2,y2,score]]，points_5 = [[x1左眼，x2左眼], [x1右眼，x2右眼], [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    # bounding_boxes=[[x1,y1,x2,y2], [x1,y1,x2,y2]]，points_5 = [[x1左眼，x2左眼], [x1右眼，x2右眼], [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    if len(bounding_boxes) < 1:
        return np.asarray([]), np.asarray([]), np.asarray([]), 0
    else:

        # 歪脸转正
        # bounding_boxes, points_5 = turn_face(bounding_boxes, points_5)

        crop = []
        det_f = bounding_boxes

        det_f[:, 0] = np.maximum(det_f[:, 0], 0)
        det_f[:, 1] = np.maximum(det_f[:, 1], 0)
        det_f[:, 2] = np.minimum(det_f[:, 2], img_size[1])
        det_f[:, 3] = np.minimum(det_f[:, 3], img_size[0])
        det_f = det_f.astype(int)


        for i in range(len(bounding_boxes)):
            temp_crop = image[det_f[i, 1]:det_f[i, 3], det_f[i, 0]:det_f[i, 2], :]
            aligned = cv2.resize(temp_crop, (image_size, image_size))
            crop.append(aligned)
        crop_image = np.stack(crop)

        points_5_crop = np.zeros(points_5.shape)
        # # 5点标记
        # f_ns = len(points_5[0])
        #
        # for xi in range(10):
        #     for fi in range(f_ns):
        #         fi_w = bounding_boxes[fi][2] - bounding_boxes[fi][0]
        #         fi_h = bounding_boxes[fi][3] - bounding_boxes[fi][1]
        #         if xi <= 4:
        #             points_5_crop[xi][fi] = ((points_5[xi][fi] - bounding_boxes[fi][0]) / fi_w) * image_size
        #         elif 4 < xi <= 9:
        #             points_5_crop[xi][fi] = ((points_5[xi][fi] - bounding_boxes[fi][1]) / fi_h) * image_size
        # points_5_crop = np.asarray(points_5_crop, dtype=int)
        return det_f, crop_image, points_5_crop, 1


def align_face(self, img, _bbox, _landmark):
    """
    used for aligning face

    Parameters:
    ----------
        img : numpy.array
        _bbox : numpy.array shape=(n,1,4)
        _landmark : numpy.array shape=(n,5,2)
        if_align : bool

    Returns:
    -------
        numpy.array

        align_face
    """
    num = np.shape(_bbox)[0]
    warped = np.zeros((num, self.img_size_list[0], self.img_size_list[1], 3))

    for i in range(num):
        warped[i, :] = preprocess(img, bbox=_bbox[i], landmark=_landmark[i], image_size=self.image_size)

    return warped


def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    assert len(image_size) == 2
    assert image_size[0] == 112
    assert image_size[0] == 112 or image_size[1] == 96
    # define desire position of landmarks
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    if image_size[1] == 112:
        src[:, 0] += 8.0

    if landmark is not None:
        assert len(image_size) == 2
        dst = landmark.astype(np.float32)

        # skimage affine
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

    #         #cv2 affine , worse than skimage
    #         src = src[0:3,:]
    #         dst = dst[0:3,:]
    #         M = cv2.getAffineTransform(dst,src)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

    return warped


def cv2_write_simsun(cv2_img, loc, text_china, char_color):
    # 设置需要显示的字体
    img_pil = Image.fromarray(cv2_img)
    draw = ImageDraw.Draw(img_pil)
    # 绘制文字信息<br># (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色颜色顺序为RGB
    draw.text(loc, text_china, font=font22, fill=char_color)
    cv2_img_new = np.array(img_pil)

    return cv2_img_new


def mark_face_points(points_lst, f_pics):
    # [[x1左眼，x2左眼], [x1右眼，x2右眼], [x1鼻子，x2鼻子], [x左嘴角，..], [x右嘴角, ..], [ y左眼，..], [y右眼，..], [y鼻子，..], [y左嘴角，..], [y右嘴角, ..]]
    cv2.line(f_pics, (points_lst[0][0], points_lst[5][0]), (points_lst[0][0], points_lst[5][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[1][0], points_lst[6][0]), (points_lst[1][0], points_lst[6][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[2][0], points_lst[7][0]), (points_lst[2][0], points_lst[7][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[3][0], points_lst[8][0]), (points_lst[3][0], points_lst[8][0]), mark_color, 2)
    cv2.line(f_pics, (points_lst[4][0], points_lst[9][0]), (points_lst[4][0], points_lst[9][0]), mark_color, 2)

    return f_pics


def mark_pic(det_lst, name_lst, pic):
    face_area_r_lst = []
    c_size = 22
    for f_i in range(len(det_lst)):
        bw = det_lst[f_i, 2] - det_lst[f_i, 0]  # (240, 248, 255)
        cv2.line(pic, (det_lst[f_i, 0], det_lst[f_i, 1]), (det_lst[f_i, 0] + int(bw * 0.20), det_lst[f_i, 1]),
                 mark_color, 2)  # 颜色是BGR顺序
        cv2.line(pic, (det_lst[f_i, 0], det_lst[f_i, 1]), (det_lst[f_i, 0], det_lst[f_i, 1] + int(bw * 0.20)),
                 mark_color, 2)
        cv2.line(pic, (det_lst[f_i, 0], det_lst[f_i, 3]), (det_lst[f_i, 0] + int(bw * 0.20), det_lst[f_i, 3]),
                 mark_color, 2)
        cv2.line(pic, (det_lst[f_i, 0], det_lst[f_i, 3]), (det_lst[f_i, 0], det_lst[f_i, 3] - int(bw * 0.20)),
                 mark_color, 2)
        cv2.line(pic, (det_lst[f_i, 2], det_lst[f_i, 1]), (det_lst[f_i, 2] - int(bw * 0.20), det_lst[f_i, 1]),
                 mark_color, 2)
        cv2.line(pic, (det_lst[f_i, 2], det_lst[f_i, 1]), (det_lst[f_i, 2], det_lst[f_i, 1] + int(bw * 0.20)),
                 mark_color, 2)
        cv2.line(pic, (det_lst[f_i, 2], det_lst[f_i, 3]), (det_lst[f_i, 2] - int(bw * 0.20), det_lst[f_i, 3]),
                 mark_color, 2)
        cv2.line(pic, (det_lst[f_i, 2], det_lst[f_i, 3]), (det_lst[f_i, 2], det_lst[f_i, 3] - int(bw * 0.20)),
                 mark_color, 2)
        # cv2.rectangle(pic, (det_lst[f_i, 0], det_lst[f_i, 1]),
        #               (det_lst[f_i, 2], det_lst[f_i, 3]), (240, 248, 255), thickness=2, lineType=8, shift=0)  # 在抓取的图片frame上画矩形
        pic = cv2_write_simsun(pic, loc=(det_lst[f_i, 0] + 8, det_lst[f_i, 1] - c_size - 8), text_china=name_lst[f_i],
                               char_color=mark_color)

        # 计算人脸占画面的面积
        area_ir = ((det_lst[f_i, 2] - det_lst[f_i, 0]) * (det_lst[f_i, 3] - det_lst[f_i, 1])) / (len(pic) * len(pic[0]))
        face_area_r_lst.append(area_ir)
    return pic, face_area_r_lst


# 创建mtcnn网络，并加载参数
print('Creating networks and loading parameters')

# gpu设置
gpu_config = tf.ConfigProto()
gpu_config.allow_soft_placement = True
gpu_config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with tf.Graph().as_default():
    sess = tf.Session(config=gpu_config)
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

# if __name__ == '__main__':
#     # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
#     capture = cv2.VideoCapture(0)
#     cv2.namedWindow("camera", 1)
#     capture_n = 0
#     timeF = 6  # 帧间隔
#
#     while True:
#         capture_n += 1
#
#         current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
#         _, frame = capture.read()
#
#         if capture_n % timeF == 0:  # 每隔6秒取一次frame实时视频数据
#
#             dets, crop_images, j = load_and_align_data(frame, 160)  # 获取人脸, 由于frame色彩空间rgb不对应问题，需统一转为灰色图片
#             if j:
#                 facenames, faceis_konwns = facenet_pre_m.imgs_get_names(crop_images)  # 获取人脸名字
#
#                 # 绘制矩形框并标注文字
#                 frame, face_areas = mark_pic(dets, facenames, frame)
#
#             cv2.imshow('camera', frame)  # 在cv2的预开窗口'camera'中显示加了框的图片帧
#             cv2.imwrite(current_time  + '_' + facenames[0] + '.jpg', frame)
#
#         if cv2.waitKey(3) == 27:  # 等待用户触发事件,等待时间为100ms， 如果在这个时间段内, 用户按下ESC(ASCII码为27), 执行if体 ,如果没有按，if函数不做处理
#             # esc键退出
#             print("esc break...")
#             break
#
#     # When everything is done, release the capture
#     capture.release()
#     cv2.destroyWindow("camera")
