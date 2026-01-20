import os
import pandas as pd
import json
from PIL import Image, ImageDraw

dataset_file = "dataset/"
image_path = "dataset/origin_images/"
data_set1 = "dataset/single-penel_qa_pair.csv"
data_set2 = "dataset/multi-panel_qa_pair.csv"
data_set3 = "dataset/reason_qa_pair.csv"

single1 = ["[Single-image - Numerical]","Single-image - Numerical"]
single2 = ["[Single-image - Graphical]","Single-image - Graphical"]
single3 = ["[Single-image - Spatial]","Single-image - Spatial"]

multi_1 = ["[Multi-image - Numerical]","Multi-image - Numerical","[Multi-image - Morphological]","Multi-image - Morphological","[Multi-image - Spatial]","Multi-image - Spatial"]
multi_2 = ["Cross-Modality relation","Multi-image - Cross-Modality relation"]

reason_1 = ["[Reason_Qualitative]"]
reason_2 = ["[Reason_Quantitative]"]

c1 = ["统计图"]
c2 = ["染色图","造影图","实物图"]
c3 = ["条带图"]
c4 = ["示意图","其他"]
def bench_stat():
    res = {
        "avg_question": 0,
        "avg_option": 0,
        "avg_evidence": 0,
        "avg_sub_num": 0,
    }
    text1 = pd.read_csv(data_set1)
    text2 = pd.read_csv(data_set2)
    text3 = pd.read_csv(data_set3)
    qa_count = 0
    for index, row in text1.iterrows():
        if os.path.exists(os.path.join(image_path, str(text1.iloc[index][0])[:-2] + "_sub.json")):

            sub_path = os.path.join(image_path, str(text1.iloc[index][0])[:-2] + "_sub.json")

            q = len(str(text1.iloc[index][2]).split())
            o = len(str(text1.iloc[index][3]).split())
            e = len(str(text1.iloc[index][5]).split())

            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            sub = len(sub_info["object"])

            res["avg_sub_num"] += sub
            res["avg_option"] += o
            res["avg_evidence"] += e
            res["avg_question"] += q
            qa_count += 1

    for index, row in text2.iterrows():
        if os.path.exists(os.path.join(image_path, str(text2.iloc[index][0])[:-2] + "_sub.json")):
            sub_path = os.path.join(image_path, str(text2.iloc[index][0])[:-2] + "_sub.json")
            q = len(str(text2.iloc[index][2]).split())
            o = len(str(text2.iloc[index][3]).split())
            e = len(str(text2.iloc[index][5]).split())

            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            sub = len(sub_info["object"])

            res["avg_sub_num"] += sub
            res["avg_option"] += o
            res["avg_evidence"] += e
            res["avg_question"] += q
            qa_count += 1

    for index, row in text3.iterrows():
        if os.path.exists(os.path.join(image_path, str(text3.iloc[index][0])[:-2] + "_sub.json")):
            sub_path = os.path.join(image_path, str(text3.iloc[index][0])[:-2] + "_sub.json")
            q = len(str(text3.iloc[index][2]).split())
            o = len(str(text3.iloc[index][3]).split())
            e = len(str(text3.iloc[index][5]).split())

            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            sub = len(sub_info["object"])

            res["avg_sub_num"] += sub
            res["avg_option"] += o
            res["avg_evidence"] += e
            res["avg_question"] += q
            qa_count += 1

    print("平均子图数",res["avg_sub_num"]/qa_count)
    print("平均选项长度",res["avg_option"] /qa_count)
    print("平均分析长度",res["avg_evidence"]/qa_count)
    print("平均问题长度",res["avg_question"]/qa_count)

def add_lists(list1, list2):
    result = []
    for i in range(len(list1)):
        result.append(list1[i] + list2[i])
    return result

def sub_image_stat():

    res = {
        "s1": [0,0,0,0],
        "s2": [0,0,0,0],
        "s3": [0,0,0,0],
        "m1": [0,0,0,0],
        "m2": [0,0,0,0],
        "r1": [0,0,0,0],
        "r2": [0,0,0,0],
    }

    text1 = pd.read_csv(data_set1)
    text2 = pd.read_csv(data_set2)
    text3 = pd.read_csv(data_set3)
    qa_count = 0

    for index, row in text1.iterrows():
        a,b,c,d = 0, 0, 0, 0
        if os.path.exists(os.path.join(image_path, str(text1.iloc[index][0])[:-2] + "_sub.json")):
            sub_path = os.path.join(image_path, str(text1.iloc[index][0])[:-2] + "_sub.json")
            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            for sub in sub_info["object"]:
                if sub["name"] in c1:
                    a += 1
                elif sub["name"] in c2:
                    b += 1
                elif sub["name"] in c3:
                    c += 1
                elif sub["name"] in c4:
                    d += 1
            tag = str(text1.iloc[index][1])

            if tag in single1:
                res["s1"] = add_lists(res["s1"],[a,b,c,d])
            elif tag in single2:
                res["s2"] = add_lists(res["s2"],[a,b,c,d])
            elif tag in single3:
                res["s3"] = add_lists(res["s3"],[a,b,c,d])

    for index, row in text2.iterrows():
        a,b,c,d = 0, 0, 0, 0
        if os.path.exists(os.path.join(image_path, str(text2.iloc[index][0])[:-2] + "_sub.json")):
            sub_path = os.path.join(image_path, str(text2.iloc[index][0])[:-2] + "_sub.json")
            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            for sub in sub_info["object"]:
                if sub["name"] in c1:
                    a += 1
                elif sub["name"] in c2:
                    b += 1
                elif sub["name"] in c3:
                    c += 1
                elif sub["name"] in c4:
                    d += 1
            tag = str(text2.iloc[index][1])

            if tag in multi_1:
                res["m1"] = add_lists(res["m1"],[a,b,c,d])
            elif tag in multi_2:
                res["m2"] = add_lists(res["m2"],[a,b,c,d])

    for index, row in text3.iterrows():
        a,b,c,d = 0, 0, 0, 0
        if os.path.exists(os.path.join(image_path, str(text3.iloc[index][0])[:-2] + "_sub.json")):
            sub_path = os.path.join(image_path, str(text3.iloc[index][0])[:-2] + "_sub.json")
            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            for sub in sub_info["object"]:
                if sub["name"] in c1:
                    a += 1
                elif sub["name"] in c2:
                    b += 1
                elif sub["name"] in c3:
                    c += 1
                elif sub["name"] in c4:
                    d += 1
            tag = str(text3.iloc[index][1])

            if tag in reason_1:
                res["r1"] = add_lists(res["r1"],[a,b,c,d])
            elif tag in reason_2:
                res["r2"] = add_lists(res["r2"],[a,b,c,d])

    print(res)
    df = pd.DataFrame(res)
    df.to_csv("sub_image_stat.csv", index=False, encoding='utf-8-sig')

def sub_image_list():
    files = os.listdir(image_path)
    a, b, c, d = 0, 0, 0, 0
    for file in files:
        if file.endswith(".json"):
            sub_path = os.path.join(image_path, file)

            with open(sub_path, 'r') as f:
                sub_info = json.load(f)

            if sub_info is not None:
                for sub in sub_info["object"]:
                    if sub["name"] in c1:
                        a += 1
                    elif sub["name"] in c2:
                        b += 1
                    elif sub["name"] in c3:
                        c += 1
                    elif sub["name"] in c4:
                        d += 1
    print( a, b, c, d)


def add_mask_to_image(img, xmin, ymin, xmax, ymax, color=(255, 0, 0), opacity=50):
    """
    在图片的指定矩形区域添加半透明纯色蒙版

    参数:
    image_path: 原图路径
    output_path: 处理后图片的保存路径
    xmin, ymin, xmax, ymax: 矩形区域坐标
    color: 蒙版颜色，默认红色(255, 0, 0)
    opacity: 透明度，0-100，默认50
    """
    # 打开原图

        # 创建一个与原图大小相同的透明图层
    mask = Image.new('RGBA', img.size, (0, 0, 0, 0))

    # 创建绘图对象
    draw = ImageDraw.Draw(mask)

    # 计算alpha值（0-255）
    alpha = int(255 * (100 - opacity) / 100)

    # 在透明图层上绘制半透明矩形
    draw.rectangle(
        [(xmin, ymin), (xmax, ymax)],
        fill=(*color, alpha)  # 添加alpha通道值
    )

    # 如果原图不是RGBA模式，转换为RGBA
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # 合并原图和蒙版
    result = Image.alpha_composite(img, mask)

    # 保存结果


    return result

#
# # 使用示例
# if __name__ == "__main__":
#     # 图片路径
#     input_image = "input.jpg"
#     output_image = "output.jpg"
#
#     # 矩形区域坐标
#     box = {
#         "xmin": 100,
#         "ymin": 150,
#         "xmax": 400,
#         "ymax": 350
#     }
#
#     # 添加蒙版（红色，50%透明度）
#     add_mask_to_image(
#         input_image,
#         output_image,
#         box["xmin"],
#         box["ymin"],
#         box["xmax"],
#         box["ymax"],
#         color=(255, 0, 0),  # 红色
#         opacity=50  # 50%透明度
#     )
#     print(f"处理完成，结果已保存到 {output_image}")


def single_image_visualize():
    output_dir = "/home/dingjunpeng/PaperMills-Dect/code/SERA-Bench/sub_image_recognize/yolo_case/"
    files = os.listdir(image_path)

    for file in files[300:500]:
        if file.endswith(".json"):
            sub_path = os.path.join(image_path, file)
            img_path = os.path.join(image_path, file[:-9]+".png")
            output_path = os.path.join(output_dir, file[:-9]+"_yolo.png")
            with open(sub_path, 'r') as f:
                sub_info = json.load(f)
            origin =  Image.open(img_path)

            #                 "xmin": 110.0,
            #                 "ymin": 577.0,
            #                 "xmax": 368.0,
            #                 "ymax": 630.0
            if sub_info is not None:
                for sub in sub_info["object"]:
                    if sub["name"] in c1:
                        origin = add_mask_to_image(
                            origin,
                            sub["bndbox"]["xmin"],
                            sub["bndbox"]["ymin"],
                            sub["bndbox"]["xmax"],
                            sub["bndbox"]["ymax"],
                            color=(253, 229, 178),  # 红色
                            opacity=50  # 50%透明度
                        )

                    elif sub["name"] in c2:
                        origin = add_mask_to_image(
                            origin,
                            sub["bndbox"]["xmin"],
                            sub["bndbox"]["ymin"],
                            sub["bndbox"]["xmax"],
                            sub["bndbox"]["ymax"],
                            color=(240, 148, 158),  # 红色
                            opacity=40  # 50%透明度
                        )
                    elif sub["name"] in c3:
                        origin = add_mask_to_image(
                            origin,
                            sub["bndbox"]["xmin"],
                            sub["bndbox"]["ymin"],
                            sub["bndbox"]["xmax"],
                            sub["bndbox"]["ymax"],
                            color=(157, 193, 231),  # 红色
                            opacity=50  # 50%透明度
                        )
                    elif sub["name"] in c4:
                        origin = add_mask_to_image(
                            origin,
                            sub["bndbox"]["xmin"],
                            sub["bndbox"]["ymin"],
                            sub["bndbox"]["xmax"],
                            sub["bndbox"]["ymax"],
                            color=(255, 255, 255),  # 红色
                            opacity=50  # 50%透明度
                        )

            origin.save(output_path)


def combine_images(grid_rows=8, grid_cols=6, img_size=(300, 300),border_width=2):
    """
    将指定目录中的图片按网格排列组合成一张大图

    参数:
    input_dir: 存放输入图片的目录
    output_path: 组合后图片的保存路径
    grid_rows: 网格行数，默认4
    grid_cols: 网格列数，默认5
    img_size: 每张图片的尺寸，默认(300, 300)
    """
    # 获取目录中所有图片文件
    input_dir = "/home/dingjunpeng/PaperMills-Dect/code/SERA-Bench/sub_image_recognize/yolo_case/"
    output_path = "/home/dingjunpeng/PaperMills-Dect/code/SERA-Bench/sub_image_recognize/combined_image.png"
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # 确保有足够的图片
    required = grid_rows * grid_cols
    if len(image_files) < required:
        raise ValueError(f"需要至少{required}张图片，但只找到{len(image_files)}张")

    # 只取需要的数量
    image_files = image_files[:required]

    # 计算组合图的尺寸
    total_width = grid_cols * img_size[0]
    total_height = grid_rows * img_size[1]

    # 创建空白的组合图
    combined_img = Image.new('RGB', (total_width, total_height))
    # 创建绘图对象用于画边框
    draw = ImageDraw.Draw(combined_img)

    # 处理并放置每张图片
    for index, img_file in enumerate(image_files):
        # 计算图片在组合图中的位置
        row = index // grid_cols
        col = index % grid_cols
        x = col * img_size[0]
        y = row * img_size[1]

        # 打开并处理图片
        try:
            with Image.open(os.path.join(input_dir, img_file)) as img:
                # 调整图片尺寸（保持比例，可能会有黑边）
                img.thumbnail(img_size)

                # 创建一个300x300的空白图片作为容器
                img_container = Image.new('RGB', img_size, (255, 255, 255))  # 白色背景

                # 计算居中放置的位置
                img_width, img_height = img.size
                offset_x = (img_size[0] - img_width) // 2
                offset_y = (img_size[1] - img_height) // 2

                # 将处理好的图片粘贴到容器中
                img_container.paste(img, (offset_x, offset_y))

                # 将容器粘贴到组合图中
                combined_img.paste(img_container, (x, y))

                # 绘制黑色边框
                # 计算边框坐标（稍微缩小一点，避免超出图片范围）
                border_x1 = x + border_width // 2
                border_y1 = y + border_width // 2
                border_x2 = x + img_size[0] - (border_width // 2)
                border_y2 = y + img_size[1] - (border_width // 2)

                draw.rectangle(
                    [(border_x1, border_y1), (border_x2, border_y2)],
                    outline="black",
                    width=border_width
                )

                print(f"已处理第{index + 1}张: {img_file}")
        except Exception as e:
            print(f"处理{img_file}时出错: {e}")

    # 保存组合图
    combined_img.save(output_path)



if __name__ == "__main__":
    bench_stat()
    Mode = {
        "single"
        "multi_trend"
        "reason"
    }
    single_image_visualize()
    combine_images()


