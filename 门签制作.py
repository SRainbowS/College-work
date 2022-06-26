"""
    作者：SRainbowS
    时间：2020/4/4
    功能：门签制作
    版本：v2.0
    使用方法：1.修改背景图路径
             2.修改宿舍文件夹路径
             3.修改保存路径
             else
             修改字体、大小
    UTF-8
"""

import os
from PIL import ImageFont, ImageDraw, Image
import openpyxl


def xuehaopipei(input_xuehao):
    # 打开excel
    wb = openpyxl.load_workbook('G:/test/xuehao.xlsx')
    ws = wb.active

    # 匹配学号
    for i in range(2, 6):
        xuehao = ws.cell(row=i, column=3).value  # 获取学号
        if xuehao == input_xuehao:
            name = ws.cell(row=i, column=4).value  # 获取name
            classname = ws.cell(row=i, column=2).value  # 获取classname
            major = ws.cell(row=i, column=6).value  # 获取major
            post = ws.cell(row=i, column=5).value  # 获取职务
            # 返回匹配信息
            output_xuehao = (name, post, major, classname)
            return output_xuehao


def main():
    # 设置背景图路径
    img = Image.open(r"G:/test/menqianbj.png")

    # 设置宿舍文件夹——注意不能有中文和空格！！！
    filepath = r'G:\\test\\1ban\\banzhang'

    # 设置保存路径
    save_path = r'G:\\test\\1ban\\1banzhang.png'

    # 总字体格式
    fontpath = "C:/Windows/Fonts/simkai.ttf"
    totalfont = ImageFont.truetype(fontpath, 125)

    # 设置人物图像大小
    S_Img_w = 441
    S_Img_h = 627

    # 第一张图片左上角位置
    start_img_x_position = 373
    start_img_y_position = 804
    # 姓名起始位置
    start_name_x_position = 850
    start_name_y_position = 842
    # 职务起始位置
    start_post_x_position = 850
    start_post_y_position = 1040
    # 专业起始位置
    start_major_x_position = 850
    start_major_y_position = 1237
    # 位置改变量
    dx = (0, 1488, 0, 1488)
    dy = (0, 0, 766, 761)

    # 写入图片、文字
    for root, dirs, files in os.walk(filepath):
        # 图片位置
        i = 0

        # 遍历文件
        for f in files:
            # 打开子图片
            img2 = Image.open(os.path.join(root, f))

            # 对子图片放缩
            icon = img2.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)

            # 贴到母图上面
            img.paste(icon, (start_img_x_position + dx[i], start_img_y_position + dy[i]))

            # 获得学号
            xuehao = int(os.path.join(root, f)[-14:-4])

            # 输出匹配值
            total = xuehaopipei(xuehao)

            # 写姓名、职务、专业
            draw = ImageDraw.Draw(img)
            # 参数:1文本的左上角所在位置(x,y),2文本内容,3字体,4文本的颜色
            draw.text((start_name_x_position + dx[i], start_name_y_position + dy[i]),
                      '姓名:' + total[0], font=totalfont, fill=(0, 0, 0))
            draw.text((start_post_x_position + dx[i], start_post_y_position + dy[i]),
                      '职务:' + total[1], font=totalfont, fill=(0, 0, 0))
            draw.text((start_major_x_position + dx[i], start_major_y_position + dy[i]),
                      '专业:' + total[2], font=totalfont, fill=(0, 0, 0))

            # 写下一个人
            i = i + 1

    # 显示最终结果
    img.show()

    # 保存
    img.save(save_path)


if __name__ == '__main__':
    main()
