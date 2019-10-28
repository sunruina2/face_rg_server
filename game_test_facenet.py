import cv2
from imutils.video import VideoStream
import numpy as np
from align import mtcnn_def as fc_server
import time
import pickle
import os
import pandas as pd


# from rg_model.model_insight_tiny import InsightPreTiny
# facenet_pre_m = InsightPreTiny()

# from rg_model.model_facenet import FacenetPre
# facenet_pre_m = FacenetPre()
# lastsave_embs0 = np.zeros(128)
# imgsize = 160
# facenet_pre_m.gen_knows_db('../facenet_files/office_face160/', '../facenet_files/embs_pkl/facenet/')

# from rg_model.model_insight_lucky import InsightPreLucky
# facenet_pre_m = InsightPreLucky()
# lastsave_embs0 = np.zeros(512)
# imgsize = 112
# # facenet_pre_m.gen_knows_db('../facenet_files/office_face160/', '../facenet_files/embs_pkl/insight_luck/')


def cos_mat(x1, x2):
    ''' Compute a distance matrix between every row of x1 and x2.'''
    assert x1.shape[1] == x2.shape[1]
    x2 = x2.transpose()
    # dot = np.dot(x1, x2).astype(int)
    dot = np.dot(x1, x2)
    x1 = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2 = np.sqrt(np.sum(np.square(x2), axis=0, keepdims=True))
    # dist = dot / np.dot(x1, x2).astype(int)
    dist = dot / np.dot(x1, x2)
    dist = 0.5 + 0.5 * dist

    return dist


def euclidean(x1, x2):
    ''' Compute a distance matrix between every row of x1 and x2.'''
    print(' Compute a distance matrix between every row of x1 and x2.')
    assert x1.shape[1] == x2.shape[1]
    x2 = x2.transpose()

    # print(x1.shape)
    # print(x2.shape)
    dot = np.dot(x1, x2).astype(int).diagonal()
    # print(dot.shape)
    x1 = np.reshape(np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True)), (x1.shape[0],))
    x2 = np.reshape(np.sqrt(np.sum(np.square(x2), axis=0, keepdims=True)), (x1.shape[0],))
    # print(x1.shape)
    # print(x2.shape)
    dist = dot / (x1 * x2)
    dist = 0.5 + 0.5 * dist

    return dist


def d_cos(v, vs=None):  # 输入需要是一张脸的v:(512,) or (512,1), knows_v:(N, 512)
    if vs is None:
        return None
    else:
        if len(vs.shape) == 1:
            vs = np.reshape(vs, (1, len(vs)))
            vs_norm = np.reshape(np.linalg.norm(vs, axis=1), (len(vs), 1))
        else:
            vs = vs
            vs_norm = np.reshape(np.linalg.norm(vs, axis=1), (len(vs), 1))

    v = np.reshape(v, (1, len(v)))  # 变为1行
    num = np.dot(vs, v.T)  # (N, 1)
    denom = np.linalg.norm(v) * vs_norm  # [|A|=float] * [|B|= reshape( (N,), (N,1) ) ] = (N, 1)
    cos = num / denom  # 余弦值 A * B / |A| * |B| 本身也是0-1之间的...

    sim = 0.5 + 0.5 * cos  # 归一化到0-1之间, (N, 1)

    # print('cos describe', max(cos), min(cos), np.mean(cos), np.var(cos))
    # print('sim describe', max(sim), min(sim), np.mean(sim), np.var(sim))
    sim = np.reshape(sim, (len(sim),))  # reshape((N,1), (N,)) 变成一维，方便后边算最大值最小值

    """
    人脸库中的照片pre_img.jpg，余弦距离参考值如下，有人脸图片cos均值在0.40842828，sim均值在 0.7042141，因此至少sim要大于0.70
    cos describe [0.99029934] [-0.07334533] 0.40842828 0.016055087
    sim describe [0.9951497] [0.46332735] 0.7042141 0.0040137717
    pre_1pic ['20190904205458_正面_024404-张佳丽'] [1] [0.9951497]

    无人脸的图片pre_bug.jpg，余弦距离参考值如下，无人脸有内容图片cos均值在0.11156807，sim均值在 0.55578405
    cos describe [0.47486433] [-0.09186573] 0.11156807 0.004270094
    sim describe [0.7374322] [0.45406714] 0.55578405 0.0010675235
    pre_1pic ['未知的同学'] [0.0] [0]

    近乎全白的图片pre_white.jpg，余弦距离参考值如下，白图cos均值在0.015752314，sim均值在 0.50787616
    cos describe [0.44681713] [-0.17200288] 0.015752314 0.00459828
    sim describe [0.7234086] [0.41399854] 0.50787616 0.0011495701
    pre_1pic ['未知的同学'] [0.0] [0]
    """

    return sim


if __name__ == '__main__':
    # pics_path = '/Users/finup/Desktop/rg/rg_game/data/Test_Data_s/'
    # pics_path = '/Users/finup/Desktop/rg/face_rg_server/data_pro/孙瑞娜/'
    pics_path = '/Users/finup/Desktop/rg/rg_game/data/Test_Data/'

    res_file = 'submission_template.csv'

    '''获取embs'''

    # from rg_model.model_facenet import FacenetPre
    # facenet_pre_m = FacenetPre()
    # imgsize = 160
    # buff_n = 250
    # batch_size = 100
    #
    # print('pic reading %s' % pics_path)
    # if os.path.isdir(pics_path):
    #     pics_name = list(os.listdir(pics_path))
    #     if '.DS_Store' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('.DS_Store')
    #     if 'all_dct.pkl' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('all_dct.pkl')
    #     if 'submission_template.csv' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('submission_template.csv')
    #     if 'submission_template.csv' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('submission_template.csv')
    #     if 'jumpjump_results.csv' in pics_name:  # 去掉不为文件夹格式的mac os系统文件
    #         pics_name.remove('jumpjump_results.csv')
    # else:
    #     pics_name = [pics_path]
    # # print(pics_name)
    # f1000_pics, f1000_names = [], []
    # all_people_embs = {}
    # with open(pics_path + 'all_dct.pkl', 'wb') as f:
    #     pickle.dump(all_people_embs, f)
    # print('saving knows pkl...', len(all_people_embs), pics_path + 'all_dct.pkl')
    # for pic_i in range(len(pics_name)):
    #     if pics_name[pic_i] != '.DS_Store' and pics_name[pic_i] != 'all_dct.pkl':
    #         # print(pics_path + pics_name[pic_i])
    #         f_pic = cv2.resize(cv2.imread(pics_path + pics_name[pic_i]), (imgsize, imgsize))
    #         f1000_pics.append(f_pic)
    #         f1000_names.append(pics_name[pic_i])
    #         if len(f1000_pics) % buff_n == 0:
    #             f1000_pics = np.asarray(f1000_pics)
    #             f1000_embs = facenet_pre_m.run_embds(f1000_pics, batch_size)
    #             f1000_dict = dict(zip(f1000_names, f1000_embs))
    #
    #             with open(pics_path + 'all_dct.pkl', 'rb') as f:
    #                 all_people_embs = pickle.load(f)
    #             print('read knows pkl...', len(all_people_embs), pics_path + 'all_dct.pkl')
    #             all_people_embs.update(f1000_dict)
    #             with open(pics_path + 'all_dct.pkl', 'wb') as f:
    #                 pickle.dump(all_people_embs, f)
    #             print('save knows pkl...', len(all_people_embs), pics_path + 'all_dct.pkl')
    #
    #             f1000_pics, f1000_names = [], []
    #             print('@@@ new add:', len(f1000_dict), 'finish :', pic_i + 1, '/', len(pics_name), '=',
    #                   np.round(pic_i / len(pics_name), 3))
    #
    # if f1000_pics != []:
    #     f1000_pics = np.asarray(f1000_pics)
    #     f1000_embs = facenet_pre_m.run_embds(f1000_pics, batch_size)
    #     f1000_dict = dict(zip(f1000_names, f1000_embs))
    #     print('@@@ last new add:', len(f1000_dict), 'finish :', pic_i + 1, '/', len(pics_name), '=',
    #           np.round(pic_i / len(pics_name), 3))
    #
    # with open(pics_path + 'all_dct.pkl', 'rb') as f:
    #     all_people_embs = pickle.load(f)
    # print('read knows pkl...', len(all_people_embs), pics_path + 'all_dct.pkl')
    # all_people_embs.update(f1000_dict)
    # with open(pics_path + 'all_dct.pkl', 'wb') as f:
    #     pickle.dump(all_people_embs, f)
    # print('save knows pkl...', len(all_people_embs), pics_path + 'all_dct.pkl')

    '''计算相似度'''

    # with open(pics_path + 'all_dct.pkl', 'rb') as f:
    #     all_people_embs = pickle.load(f)
    #     print('read knows pkl...', len(all_people_embs), pics_path + 'all_dct.pkl')
    #
    # k = np.asarray(list(all_people_embs.keys()))
    # k_idx_dct = dict(zip(k, list(range(len(k)))))
    # v = np.asarray(list(all_people_embs.values()))
    # print('kv shape', k.shape, v.shape)
    # sim_mat = cos_mat(v, v)
    # v = []
    # print('simmat ', sim_mat.shape)
    #
    # res = pd.read_csv(pics_path + res_file, header=None)
    # res = np.asarray(res)
    #
    # not_found_n = 0
    # for i in range(res.shape[0]):
    #     pair = res[i][0].split(':')
    #     pic1, pic2 = pair[0], pair[1]
    #     try:
    #         res[i, 1] = sim_mat[k_idx_dct[pic1]][k_idx_dct[pic2]]
    #     except:
    #         not_found_n += 1
    #         res[i, 1] = 0.45
    #         print(pic1, pic2, 'not found!')
    #     if i % 1000:
    #         print('finish', np.round(i / res.shape[0], 4), 'not_found_n', not_found_n)
    #
    # print(res[1].describe())
    # pd.DataFrame(res).to_csv(pics_path + 'jumpjump_results.csv', header=None, index=False)
