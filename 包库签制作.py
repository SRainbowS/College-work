"""
    作者：SRainbowS
    时间：2020/4/2
    功能：宝库签
    版本：v2.0
    使用方法：1.更新excel表
             2.修改保存路径
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

    # 设置需要显示的字体
    fontpath = "C:/Windows/Fonts/simkai.ttf"
    # 字体_大小
    namefont = ImageFont.truetype(fontpath, 420)
    classnamefont = ImageFont.truetype(fontpath, 130)

    # 绘制文字信息
    for i in range(2, 6):
        name = ws.cell(row=i, column=4).value  # 获取第2行
        classname = ws.cell(row=i, column=2).value  # 获取第1行

        # 输入背景图片
        bk_img = cv2.imread("G:/test/baokubj.png")
        img_pil = Image.fromarray(bk_img)
        draw = ImageDraw.Draw(img_pil)

        # 参数:1文本的左上角所在位置(x,y),2文本内容,3字体,4文本的颜色
        draw.text((390, 450), name, font=namefont, fill=(0, 0, 0))  # 姓名
        draw.text((570, 200), classname, font=classnamefont, fill=(0, 0, 0))  # 班级
        bk_img = np.array(img_pil)

        # 保存路径
        cv2.imencode('.png', bk_img)[1].tofile("G:/test/baokuqian_all/" + name + ".png")


if __name__ == '__main__':
    main()
