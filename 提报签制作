"""
    作者：SRainbowS
    时间：2020/4/2
    功能：提报签
    版本：v1.0
    使用方法：1.更新excel表,保存路径
             2.修改背景路径
             3.修改保存路径
             else
             修改字体、大小
    UTF-8
"""

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import openpyxl


def main():
    # 打开工作簿
    wb = openpyxl.load_workbook('G:/test/xuehao.xlsx')
    ws = wb.active

    # 背景图片位置
    background_position = "G:/test/tibaoqianbj.png"

    # 保存路径——注意最后有'/'
    save_path = "G:/test/tibaoqian_all/"

    # 设置需要显示的字体
    fontpath = "C:/Windows/Fonts/simkai.ttf"

    # 设置位置
    name_x_position = 1368
    name_y_position = 1041
    classname_x_position = 1710
    classname_y_position = 1868
    name_size = 590
    classname_size = 250

    # 字体_大小
    namefont = ImageFont.truetype(fontpath, name_size)
    classnamefont = ImageFont.truetype(fontpath, classname_size)

    # 绘制文字信息
    for i in range(2, 6):                           # 设置从多少到多少
        name = ws.cell(row=i, column=4).value       # 第4列
        classname = ws.cell(row=i, column=7).value  # 第7列

        # 输入背景图片
        bk_img = cv2.imread(background_position)
        img_pil = Image.fromarray(bk_img)
        draw = ImageDraw.Draw(img_pil)

        # 参数:1文本的左上角所在位置(x,y),2文本内容,3字体,4文本的颜色
        draw.text((name_x_position, name_y_position), name, font=namefont, fill=(0, 0, 0))  # 姓名
        draw.text((classname_x_position, classname_y_position), classname, font=classnamefont, fill=(0, 0, 0))  # 班级
        bk_img = np.array(img_pil)

        # 保存路径
        cv2.imencode('.png', bk_img)[1].tofile(save_path + name + ".png")


if __name__ == '__main__':
    main()
