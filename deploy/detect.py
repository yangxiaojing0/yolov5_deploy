"""Perform test request"""
import json
import pprint
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from color import Colors

LOCAL_DETECTION_URL = "http://222.28.54.95:8801/v1/object-detection/yolov5s"
CLOUD_DETECTION_URL = "http://222.28.54.95:8801/v1/object-detection/yolov5s"


def draw_box_to_img(img,result):
    '''
    绘制检测结果
    result：list. 相应的response
    '''
    if type(img)==(str or Path):
        image = cv2.imread(img)
    else:
        image =img
    
    for box_det in result:
        x1, y1, x2, y2 = map(int, [box_det['xmin'],box_det['ymin'],box_det['xmax'],box_det['ymax']])
        # 在图像上绘制矩形框，并添加类别标签
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框线
        # colors = Colors()
        # cv2.rectangle(image, (x1, y1), (x2, y2), color=colors(c, True), 2)  # todo: 自定义框线颜色
        cv2.putText(
            image,
            f"{box_det['name']} {box_det['confidence']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )  # 红色类别标签
    return image
    
def detection_request(test_image,show_img=False,save_json_dir=None,save_img_dir=None, ret=None):
    '''
    单张图片请求检测
    test_image：str or <class 'numpy.ndarray'>。用于检测的图像
    show_img：True or false. opencv展示图片
    save_json_dir: None or dir_str. 保存检测结果json
    save_img_dir：None or dir_str. 保存检测绘制的图片
    ret：None or str. 用于摄像头检测，用作保存的图片名
    '''
    if type(test_image)==str:
        image_data = open(test_image, "rb").read() # bytes
        test_image_name=Path(test_image).name
        test_stem=Path(test_image).stem
        
    else:# <class 'numpy.ndarray'>
        image_cv2 = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        # 将图像编码为字节形式
        success, encoded_image = cv2.imencode('.jpg', image_cv2)
        if not success:
            raise Exception("Error encoding image")
        
        # 转换为字节类型
        image_data = encoded_image.tobytes()
        
        test_image_name=f'{ret}.jpg'
        test_stem=f'{ret}'
        
    response = requests.post(LOCAL_DETECTION_URL, files={"image": image_data}).json()
    
    result={test_image_name:response}
    pprint.pprint(result)
    image=draw_box_to_img(test_image,response)
    
    if save_json_dir:
        json_name=str(Path(save_json_dir)/(test_stem+'.json'))
        with open(json_name,'w',encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    
    if show_img:
        cv2.imshow("Image with boxes", image)
        cv2.waitKey(0)  # 等待；0：按键。数字：毫秒级
        # cv2.waitKey(10000)
        cv2.destroyAllWindows()  # 销毁所有窗口
    if save_img_dir:
        image_name=test_image.split('/')[-1]
        save_path=str(Path(save_img_dir)/image_name)
        cv2.imwrite(save_path, image)
    return image


def camera_detect(save_path, interval=5,detect=True, show_img=True, save_img=True,show_time=False):
    '''
    调用摄像头检测
    save_path：str. 结果保存路径
    interval：int. 采样间隔
    detect：true or false. 是否进行检测。调用detection_request()函数
    show_img：是否实时展示画面
    save_img：是否保存结果
    show_time：是否在图像上显示采样时间
    '''
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 定义计数器和时间戳变量
    counter = 0
    timestamp = 0

    while True:
        # 逐帧读取视频序列
        ret, frame = cap.read()
        if not ret:
            break
        # 每a帧读取一次
        counter += 1
        if counter == interval:
            # 读取当前时间戳
            current_time = time.time()
            # 显示帧和时间戳信息
            if show_time:
                cv2.putText(
                    frame,
                    f"Frame: {ret}, Time: {timestamp:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
            
            # 实时检测
            if detect:
                # print(type(frame)) # <class 'numpy.ndarray'>
                frame=detection_request(frame,save_json_dir=None,save_img_dir=None,show_img=None,ret=ret)
                
            if show_img:
                cv2.imshow("Frame", frame)
                
            if save_img:
                # 保存帧和时间戳到文件或数据库等
                cv2.imwrite(f"{save_path}/img{current_time}.png", frame)
                
            # 重置计数器和时间戳变量
            counter = 0
            timestamp = current_time
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 停止采样
            break

    # 释放摄像头和窗口资源qq
    cap.release()
    cv2.destroyAllWindows()       


def print_page(URL):
    if URL==LOCAL_DETECTION_URL:
        print('**********************************************')
        print('*****                                    *****')
        print('                将在本地进行检测                ')
        print('*****                                    *****')
        print('**********************************************')
        
    elif URL==CLOUD_DETECTION_URL:
        print('**********************************************')
        print('*****                                    *****')
        print('                将在服务器进行检测               ')
        print('*****                                    *****')
        print('**********************************************')
    else:
        raise ValueError('URL错误')
        
    
if __name__ == '__main__':
    '''单张图片检测'''
    # test_image = "test_img/bus.jpg"
    # save_json_dir='save_json'
    # save_img_dir='save_img'
    # Path(save_json_dir).mkdir(exist_ok=True,parents=True)
    # Path(save_img_dir).mkdir(exist_ok=True,parents=True)
    
    # detection_request(test_image,save_json_dir=save_json_dir,save_img_dir=save_img_dir,show_img=True)
    
    '''本地camera检测'''
    print_page(LOCAL_DETECTION_URL)
    save_path = r"./camera"  # 保存路径
    Path(save_path).mkdir(exist_ok=True,parents=True)
    a = 2  # 采样间隔
    camera_detect(save_path, a, detect=True, show_img=True,save_img=True,show_time=True)
    