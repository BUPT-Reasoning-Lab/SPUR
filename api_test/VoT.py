#测试代码
import copy
import json
import asyncio
import aiolimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import os
import logging
from typing import Dict
import base64
import pandas as pd
import re
from openai import OpenAI
SYSTEM_INSTRUCTION = (
    " You are a helpful assistant specialized in analyzing academic images. Please answer this question."
    " Answer with the option's letter from the given choices."
    " Visualize the state after each reasoning step."
    " Make sure to follow this output format strictly:"
    " <ANSWER> the correct answer (A or B or C or D or E) of question here </ANSWER>"
    " <EVIDENCE> the explain of your answer here </EVIDENCE>"
    )


class Config:
    dataset_file: str = "dataset/"  # 数据集文件路径
    image_path :str = "dataset/origin_images/"
    # 输出结果保存的根目录
    outcome_dir: str = "results/multi_results/"

    # 模型列表

    # model_name: str = "google/gemini-2.5-pro-preview"  # 模型名称
    # model_name: str = "openai/o4-mini-high"  # 模型名称（有地区限制）
    # model_name: str = "openai/gpt-4o"  # 模型名称（有地区限制）
    # model_name: str = "meta-llama/llama-4-maverick"  # 模型名称
    # model_name: str = "anthropic/claude-3.7-sonnet:thinking"  # 模型名称
    # model_name: str = "google/gemma-3-27b-it"  # 模型名称
    # model_name: str = "qwen/qwen3-32b"  # 模型名称（不支持视觉输入）
    model_name: str = "qwen/qwen2.5-vl-72b-instruct"  # 视觉语言模型（推荐）
    # model_name: str = "qwen/qwen2.5-vl-32b-instruct"  # 视觉语言模型
    # model_name: str = "opengvlab/internvl3-14b"  # 视觉语言模型
    # model_name: str = "openai/gpt-5.1"
    # model_name: str = "x-ai/grok-4.1-fast"
    # model_name: str="qwen/qwen3-vl-30b-a3b-thinking"
    # model_name: str="qwen/qwen3-vl-30b-a3b-instruct"
    # model_name: str="z-ai/glm-4.5v"
    rpm: int = 20  # 每分钟请求数限制
    max_no_improve_round_count: int = 10  # 最大连续未成功请求的轮数
    process_count: int = 3 # 总共处理的数据项数量（用于测试） #-1表示处理所有数据
    # openrouter
    client: AsyncOpenAI = AsyncOpenAI(
        api_key="",
        base_url="https://openrouter.ai/api/v1",
    )

    max_input_images: int = 10  # 最大输入图片数量

    @staticmethod
    def setup_logging(output_dir: str):
        """设置日志配置"""
        log_file = os.path.join(output_dir, "execution.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)


def create_output_dirs(config: Config) -> str:
    """创建以时间戳、处理数据量命名的输出目录"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 提取模型名称的最后一部分
    model_name_part = config.model_name.split("/")[-1]
    output_dir = os.path.join(
        config.outcome_dir,
        config.dataset_file.split("/")[-1].split(".")[0],
        model_name_part,
        f"{timestamp}_process_count_{config.process_count}",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rounds_outcome"), exist_ok=True)
    return output_dir


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"图片编码错误 {image_path}: {str(e)}")
        return ""


async def process_one_item(
    data: Dict, config: Config, limiter: aiolimiter.AsyncLimiter
) -> Dict:
    """负责处理单个数据项，包含速率限制。针对本次数据集任务改进"""
    async with limiter:
        try:
            # 准备消息内容
            messages = []

            # 添加系统消息
            messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})
            # 准备用户消息内容和图片
            user_content = []


            user_content.append({"type": "text", "text": data["QUESTION"] + "\n Options:"+ data["OPTION"]})

            # 添加图片内容
            image = os.path.join(config.image_path, data["QUESTION_INDEX"][:-2] + ".png")
            base64_image = encode_image_to_base64(image)
            if base64_image:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    }
                )

            # 添加用户消息
            if user_content:
                messages.append({"role": "user", "content": user_content})

            # 调用OpenAI API执行任务
            # 声明一个response变量
            response = None
            response = await config.client.chat.completions.create(
                model=config.model_name,
                messages=messages,
            )
            pattern = fr'<ANSWER>(.*?)</ANSWER>'
            matches = re.findall(pattern, response.choices[0].message.content, re.DOTALL)
            if len(matches) > 0:
                answer = matches[0]
            else:
                answer = "No choice"

            data["is_success"]= True
            data["simple_answer"]= str(answer).strip()
            if str(answer) == data["ANSWER"]:
                data["final"] = True
            else:
                data["final"] = False
            data["raw_response"]= response.choices[0].message.content
        except Exception as e:
            if response is not None:
                error_message = str(e) + "\t" + response.model_dump_json()
            else:
                error_message = str(e)
            logging.error(f"处理数据时出错: {error_message}")

            data["is_success"] = False

        return data


# 负责一轮处理
async def process_round(
    config: Config,
    limiter: aiolimiter.AsyncLimiter,
    pending_items_map: Dict,
    final_results: Dict,
) -> Dict:
    # 创建任务列表
    tasks = []
    for index, item in pending_items_map.items():
        tasks.append(process_one_item(item, config, limiter))

    current_round_results = await tqdm_asyncio.gather(*tasks)

    for i, result in enumerate(current_round_results):
        index = list(pending_items_map.keys())[i]
        final_results[index] = result

    return final_results


# 异步调用的主函数
async def main_async(mode):
    Mode = {
        "single": "single-panel_qa_pair.json",
        "multi_trend": "multi-panel_trend_qa_pair.json",
        "reason": "reason_qa_pair.json",
    }

    # 基础配置
    config = Config()
    output_dir = create_output_dirs(config)
    config.setup_logging(output_dir)

    json_path = os.path.join(config.dataset_file,Mode[mode])
    round_outcome_dir = os.path.join(output_dir, "rounds_outcome")
    limiter = aiolimiter.AsyncLimiter(1, 60.0 / config.rpm)

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if config.process_count > 0:
        dataset = dataset[: config.process_count]

    final_results = copy.deepcopy(dataset)
    pending_items_map = {i: dataset[i] for i in range(len(dataset))}
    round_count = 0

    # 主循环，每次循环代表一轮，会处理所有没有处理（或处理失败）的数据
    while (
        len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count
    ):
        round_count += 1
        logging.info(f"第{round_count}轮，剩余未处理数据: {len(pending_items_map)}")

        # 处理一轮
        final_results = await process_round(
            config, limiter, pending_items_map, final_results
        )
        with open(
            os.path.join(round_outcome_dir, f"round_{round_count}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        pending_items_map = {
            i: final_results[i]
            for i in range(len(final_results))
            if not final_results[i]["is_success"]
        }
        if len(pending_items_map) == 0:
            break

    # 生成文件名基础部分（与CSV保持一致）
    filename_base = mode + "_" + config.model_name.split("/")[-1] + "_vot_results"
    
    # 保存JSON文件
    json_filename = filename_base + ".json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    logging.info(f"JSON结果已保存到: {json_path}")

    # 保存CSV文件
    df = pd.DataFrame(final_results)
    csv_filename = filename_base + ".csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"CSV结果已保存到: {csv_path}")

if __name__ == "__main__":
    # Mode = {
    #     "single"
    #     "multi_trend"
    #     "reason" 
    # }

    asyncio.run(main_async("multi_trend"))
