import json
import os.path
import time
from openai import OpenAI, AzureOpenAI
import base64
import pandas as pd
import re

# 大图存放路径  和  文本信息存放路径
image_path = "origin_images/"
text_path = "origin_text.xlsx"


client = OpenAI(
    api_key="",
    base_url="",
)


def format_time(seconds):
    # 计算小时、分钟和秒
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 返回格式化的字符串
    return f"{int(hours)}小时{int(minutes)}分{int(seconds)}秒"


def encode_image(image_path):
    """将图像编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"图像编码错误 {image_path}: {str(e)}")
        return None




def get_4o_response(instruction,image_path,caption,rs):


    """获取GPT基于图像的答案和解释"""
    try:
        base64_image = encode_image(image_path)
        if not base64_image:
            return None
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please generate questions under system instruction."
                            + "[Image caption]: " + str(caption) + "[Related sentence]: " + str(rs)  # user_prompt

                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens= 3000  # 增加token数以容纳解释
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"获取GPT回应时出错: {str(e)}")
        return None


def qa_pair_filter(text):

    tags = ['QUESTION_INDEX', 'QUESTION_TAG', 'QUESTION', 'OPTION', 'ANSWER', 'EVIDENCE']
    paragraphs = []

    # 首先尝试按QUESTION_INDEX分割文本
    index_pattern = r'<QUESTION_INDEX>(\d+)</QUESTION_INDEX>'
    index_matches = list(re.finditer(index_pattern, text))

    if not index_matches:
        return None

    for i, match in enumerate(index_matches):
        start_pos = match.start()
        end_pos = index_matches[i + 1].start() if i < len(index_matches) - 1 else len(text)

        paragraph_text = text[start_pos:end_pos]
        result = {}

        # result['QUESTION_INDEX'] = [match.group(1)]

        for tag in tags:
            pattern = fr'<{tag}>(.*?)</{tag}>'
            matches = re.findall(pattern, paragraph_text, re.DOTALL)
            result[tag] = [match.strip() for match in matches]

        paragraphs.append(result)

    return paragraphs

if __name__ == "__main__":
    qa_res = []
    text = pd.read_excel(text_path)
    # Mode = "single"
    # Mode = "multi_trend"
    Mode = "reason"

    json_path = Mode + "_qa_pair.json"
    csv_path = Mode + "_qa_pair.csv"
    prompt = {
        "single": "single.txt",
        "multi_trend": "multi_trend.txt",
        "reason": "reason.txt",
    }
    # temp = pd.read_csv(csv_path)
    # old_qa = temp.to_dict("records")
    # count = 0
    # with open(prompt[Mode], 'r') as file:
    #     instruction = file.read()
    #
    # # text.reindex(index=text.index[::-1])
    #
    # for qa in old_qa[:]:
    #     if qa["QUESTION_TAG"] == "[Reason_Quantitative]":
    #         if count%3 == 0:
    #             try:
    #                 fig_name = qa["QUESTION_INDEX"][:-2] + ".png"
    #                 print(fig_name)
    #                 origin = text.loc[text['fig_name'] == fig_name]
    #                 origin = origin.to_dict("records")[0]
    #                 print(origin)
    #                 fig_id = str(origin['fig_name']).split(".png")[0]
    #                 image = os.path.join(image_path, fig_name)
    #                 caption = origin["caption"]
    #                 rs = origin["related_sentences"]
    #
    #                 print(image)
    #                 print(caption)
    #                 raw_qa = get_4o_response(instruction, image, caption, rs)
    #                 print(raw_qa)
    #                 parsed_data = qa_pair_filter(raw_qa)
    #                 # count = count + len(parsed_data)
    #                 # print(len(parsed_data))
    #                 if parsed_data:
    #                     for i, paragraph in enumerate(parsed_data):
    #                         # print(f"\n段落 {i + 1}:")
    #                         qa_id = fig_id + "_" + str(paragraph['QUESTION_INDEX'][0])
    #                         paragraph['QUESTION_INDEX'] = qa_id
    #                         paragraph['QUESTION_TAG'] = paragraph['QUESTION_TAG'][0]
    #                         paragraph['QUESTION'] = paragraph['QUESTION'][0]
    #                         paragraph['OPTION'] = paragraph['OPTION'][0]
    #                         paragraph['ANSWER'] = paragraph['ANSWER'][0]
    #                         paragraph['EVIDENCE'] = paragraph['EVIDENCE'][0]
    #
    #                         print(qa_id)
    #                         print(paragraph['QUESTION_TAG'])
    #                         qa_res.append(paragraph)
    #             except:
    #                 print("one error skip")
    #         else:
    #             qa_res.append(qa)
    #
    #         count = count + 1
    #
    #     else:
    #         qa_res.append(qa)
    #
    #
    #
    # with open("new_" + json_path, 'w', encoding='utf-8') as f:
    #     json.dump(qa_res, f, ensure_ascii=False, indent=2)
    #
    # df = pd.DataFrame(qa_res)
    # df.to_csv("new_"+ csv_path, index=False, encoding='utf-8-sig')
    count = 0
    with open(prompt[Mode], 'r') as file:
        instruction = file.read()

    text.reindex(index=text.index[::-1])

    for index, row in text.iterrows():
        if count >= 1000:
            break

        print(index)

        try:
            fig_id = str(text.iloc[index][1].split(".png")[0])
            image = os.path.join(image_path ,text.iloc[index][1])
            caption = str(text.iloc[index][2])
            rs = str(text.iloc[index][3])
            # print(image)

            raw_qa = get_4o_response(instruction,image,caption,rs)
            # print(raw_qa)
            parsed_data = qa_pair_filter(raw_qa)
            count = count + len(parsed_data)
            print(len(parsed_data))
            if parsed_data:
                for i, paragraph in enumerate(parsed_data):
                    # print(f"\n段落 {i + 1}:")
                    qa_id = fig_id + "_" + str(paragraph['QUESTION_INDEX'][0])
                    paragraph['QUESTION_INDEX'] = qa_id
                    paragraph['QUESTION_TAG'] = paragraph['QUESTION_TAG'][0]
                    paragraph['QUESTION'] = paragraph['QUESTION'][0]
                    paragraph['OPTION'] = paragraph['OPTION'][0]
                    paragraph['ANSWER'] = paragraph['ANSWER'][0]
                    paragraph['EVIDENCE'] = paragraph['EVIDENCE'][0]

                    print(qa_id)
                    print(paragraph['QUESTION_TAG'])
                    qa_res.append(paragraph)
        except:
            print("one error skip")

        # time.sleep(1)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_res, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(qa_res)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
