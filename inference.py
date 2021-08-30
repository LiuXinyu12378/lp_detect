# Compared to fh0.py
# fh02.py remove the redundant ims in model input
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retina import Retina
from utils.box_utils import decode, decode_landm
import time
import torchvision
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from imutils import paths
from main_model import build_model
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from numpy import random

parser = argparse.ArgumentParser(description='RetinaPL')

parser.add_argument('-m', '--trained_model', default='weight/mobilenet0.25_epoch_20_ccpd.pth',type=str, help='Trained state_dict file path to open')
# parser.add_argument('-m', '--trained_model', default='weight/Resnet50_Final.pth',type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.01, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('-image', default='image_test/image/lp1.jpg', help='test image path')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')

args = parser.parse_args()
provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X','Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


resume_file = 'weight/CNNLP_main_98.79.pth'

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0].clamp_(0, img_shape[1])  # x1
    boxes[1].clamp_(0, img_shape[0])  # y1
    boxes[2].clamp_(0, img_shape[1])  # x2
    boxes[3].clamp_(0, img_shape[0])  # y2


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=device ) #lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = Retina(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
#     print(net)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    weights = './weight/best.pt'
    vehical_model = attempt_load(weights, map_location=device)
    imgsz = 640
    imgsz = check_img_size(imgsz, s=vehical_model.stride.max())  # check img_size
    source = 'image_test/image/'
    dataset = LoadImages(source, img_size=imgsz)
    
    names = vehical_model.module.names if hasattr(vehical_model, 'module') else vehical_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = vehical_model(img) if device.type != 'cpu' else None  # run once
    cout =0
    for path, img, im0s, vid_cap in dataset:   
        img = torch.from_numpy(img).to(device)    
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = vehical_model(img, augment=args.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
        im0 = im0s
        
        for i, det in enumerate(pred):  # detections per image
#             s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            resut_lp_chines= []
            coor_list = []
            if len(det):
                
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#              
                for *xyxy, conf, cls in reversed(det):
#                     print(conf)
#                     print(cls)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    line_thickness=3
                    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#                     color = color or [random.randint(0, 255) for _ in range(3)]
                    color = [random.randint(0, 255) for _ in range(3)]
                    color = (255, 255, 0)
#                     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#                     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    x1,y1,x2,y2 =  int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    sx, sy, ex, ey = x1,y1,x2,y2 
                    cv2.rectangle(im0, (x1,y1), (x2,y2), color, thickness=tl, lineType=cv2.LINE_AA)
                    w = int(x2 - x1 + 1.0)
                    h = int(y2 - y1 + 1.0)
                    img_box = np.zeros((h, w, 3))
                    im_height, im_width, _ = im0.shape
                    if (y1-3)<=0 or (x1-3) <=0:
                        y1, x1 = 0,0
                        img_box = im0[y1+1:y2, x1+1:x2, :]
                    elif (y2+3) >=im_height or  (x2+3)>=im_width:
                        y2, x2 = im_height,im_width
                        img_box = im0[y1:y2-1, x1:x2-1, :]
                    elif (y1-3)>0 and (x1-3)>0 and (y2 + 3)<im_height and  (x2 + 3)<im_width:
                        img_box = im0[y1-3:y2 + 3, x1-3:x2 + 3, :]
                    else:
                        img_box = im0[y1:y2 , x1:x2 , :]
                    img_videcal_raw = img_box  # 这是每一辆车辆的
                    img = cv2.resize(img_box, (720,1160))
                    img_raw = img
                    img = np.float32(img)
                    resize = 1
                    

                    im_height, im_width, _ = img.shape
                    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    img -= (104, 117, 123)
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = img.to(device)
                    scale = scale.to(device)

                    tic = time.time()
                    loc, conf, landms = net(img)  # forward pass
                    # print('net forward time: {:.4f}'.format(time.time() - tic))

                    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                    priors = priorbox.forward()
                    priors = priors.to(device)
                    prior_data = priors.data
                    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                    boxes = boxes * scale / resize
                    boxes = boxes.cpu().numpy()

                    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

                    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                           img.shape[3], img.shape[2],
                                           img.shape[3], img.shape[2]])
                    scale1 = scale1.to(device)
                    landms = landms * scale1 / resize
                    landms = landms.cpu().numpy()

                    # ignore low scores
                    inds = np.where(scores > args.confidence_threshold)[0]
                    boxes = boxes[inds]
                    landms = landms[inds]
                    scores = scores[inds]

                    # keep top-K before NMS
                    order = scores.argsort()[::-1][:args.top_k]
                    boxes = boxes[order]
                    landms = landms[order]
                    scores = scores[order]

                    # do NMS
                    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = py_cpu_nms(dets, args.nms_threshold)
                    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                    dets = dets[keep, :]
                    landms = landms[keep]

                    # keep top-K faster NMS
                    dets = dets[:args.keep_top_k, :]
                    landms = landms[:args.keep_top_k, :]

                    dets = np.concatenate((dets, landms), axis=1)
                    # print('priorBox time: {:.4f}'.format(time.time() - tic))
                    # print('save image')
                    # show image
                    if args.save_image:
                        for b in dets:
                            if b[4] < args.vis_thres:
                                continue
                            text = "{:.4f}".format(b[4])
                            print(text)
                            b = list(map(int, b))

                            cx = b[0]
                            cy = b[1] + 12
        

                            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

                            w = int(x2 - x1 + 1.0)
                            h = int(y2 - y1 + 1.0)
                            img_box = np.zeros((h, w, 3))
                            if (y1-3)<=0 or (x1-3) <=0:
                                y1, x1 = 0,0
                                img_box = img_raw[y1+1:y2, x1+1:x2, :]
                            elif (y2+3) >=im_height or  (x2+3)>=im_width:
                                y2, x2 = im_height,im_width
                                img_box = img_raw[y1:y2-1, x1:x2-1, :]
                            elif (y1-3)>0 and (x1-3)>0 and (y2 + 3)<im_height and  (x2 + 3)<im_width:
                                img_box = img_raw[y1-3:y2 + 3, x1-3:x2 + 3, :]
                            else:
                                img_box = img_raw[y1:y2 , x1:x2 , :]


                            new_x1, new_y1 = b[9] - x1, b[10] - y1
                            new_x2, new_y2 = b[11] - x1, b[12] - y1
                            new_x3, new_y3 = b[7] - x1, b[8] - y1
                            new_x4, new_y4 = b[5] - x1, b[6] - y1

                            points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
                            points2 = np.float32([[0, 0], [240, 0], [0, 80], [240, 80]])

                   
                            M = cv2.getPerspectiveTransform(points1, points2)

                            # 实现透视变换转换
                            processed = cv2.warpPerspective(img_box, M, (240, 80))

         
                            # cv2.imwrite('bbox.jpg', processed)


                            model_conv = build_model(96, 0.5)
                            model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
                            model_conv.load_state_dict(torch.load(resume_file,map_location=torch.device("cuda")))
                            model_conv = model_conv.cuda()
                            model_conv.eval()

                            processed_img = processed.astype('float32')
                            processed_img = cv2.resize(processed_img, (94,24))
                            processed_img /= 255.0
                            processed_img = processed_img.transpose(2, 0, 1)
                            processed_img = torch.from_numpy(processed_img).unsqueeze(0)

                            y_pred = model_conv(processed_img)
                            outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
                            labelPred = [t[0].index(max(t[0])) for t in outputY]
                            lpn = provinces[labelPred[0]]+alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]
                            print('识别结果',lpn)
                            resut_lp_chines.append(lpn)

                            scale_x = (ex - sx) / 720
                            scale_y = (ey - sy) / 1160
                            b[0] = b[0] * scale_x + sx
                            b[1] = b[1] * scale_y + sy
                            b[2] = b[2] * scale_x + sx
                            b[3] = b[3] * scale_y + sy

                            

                            cv2.rectangle(im0, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)  # b 是针对 img, 也就是 车辆resize后的720x1160后的坐标
                            coor_list.append((int(b[0]), int(b[1])-40))
                            
    
#                                             # landms
#                             cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
#                             cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
#                             # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
#                             cv2.circle(img_raw, (b[9], b[10]), 1, (0, 255, 0), 4)
#                             cv2.circle(img_raw, (b[11], b[12]), 1, (255, 0, 0), 4)

                pilImg=Image.fromarray(cv2.cvtColor(im0,cv2.COLOR_BGR2RGB))
                draw=ImageDraw.Draw(pilImg)
                font = ImageFont.truetype('SimHei.ttf', 20, encoding="utf-8")
                for  j in range(len(resut_lp_chines)):
                    print(coor_list[j])
                    draw.text(coor_list[j], resut_lp_chines[j], (255, 0, 0), font=font)
                cv2charimg = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
#             #                 save_path = 'baidu_test/'
#             #                 dstFileName=save_path + name
                print(cout)
                cout +=1
                dstFileName = 'image_test/test/'+str(cout)+'.jpg'
                cv2.imwrite(dstFileName, cv2charimg)
                print('图片保存地址',dstFileName)
                    

                



