import copy
import json
import asyncio
import aiolimiter
# form openai import AsyncOpenAI # 如果不使用 OpenAI 的模型，可以注释掉或保留
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime
import os
import sys
import logging
from typing import Dict
import base64
import pandas as pd
import re
import tempfile  # [新增] 用于创建临时文件
import shutil  # [新增] 用于清理文件
import argparse
import yaml
from LLM.LLM_state import set_api_client, set_vic_api_client
from LLM.LLM_route import API_client
from Vic.Vic_main import Vic_async

# [新增] 添加src目录到路径，确保可以正确导入utils模块
# 获取当前文件所在目录的父目录（即src目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # utils的父目录是src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


# 系统指令可以保留，作为 prompt 的一部分传入
SYSTEM_INSTRUCTION = (
    " You are a helpful assistant specialized in analyzing academic images. Please answer this question."
    " Answer with the option's letter from the given choices."
    " Make sure to follow this output format strictly:"
    " <ANSWER> the correct answer (A or B or C or D or E) of question here </ANSWER>"
    " <EVIDENCE> the explain of your answer here </EVIDENCE>"
)


# # [新增] 一个简单的类，用于模拟 argparse 的 args 对象
# class AgentArgs:
#     def __init__(self, decision_path, image_path, query, api_key=None, client=None, model_name=None,
#                  detail_log_file=None):
#         self.decision_path = decision_path
#         self.image_path = image_path
#         self.query = query
#         self.api_key = api_key  # Google API key (当使用Google API时)
#         self.client = client  # OpenAI client (当使用OpenRouter时)
#         self.model_name = model_name  # 模型名称 (当使用OpenRouter时)
#         self.detail_log_file = detail_log_file  # 详细日志文件句柄
#         # API配置参数（可选，会在使用时设置）
#         self.max_tokens = 8192
#         self.timeout = 120
#         self.max_retries = 5

# 模型列表（选择要测试的模型）
# model_name= "google/gemini-2.5-pro-preview"  # 模型名称
# model_name = "openai/o4-mini-high"  # 模型名称（有地区限制）
# model_name= "openai/gpt-4o"  # 模型名称（有地区限制）
# model_name = "meta-llama/llama-4-maverick"  # 模型名称
# model_name = "anthropic/claude-3.7-sonnet:thinking"  # 模型名称
# model_name = "google/gemma-3-27b-it"  # 模型名称
# model_name = "qwen/qwen2.5-vl-72b-instruct"  # 视觉语言模型（推荐）
# model_name = "qwen/qwen2.5-vl-32b-instruct"  # 视觉语言模型
# model_name = "opengvlab/internvl3-14b"  # 视觉语言模型
# model_name = "openai/gpt-5.1"
# model_name = "x-ai/grok-4.1-fast"
# model_name = "qwen/qwen3-vl-30b-a3b-thinking"
model_name = "qwen/qwen3-vl-30b-a3b-instruct"
# model_name = "z-ai/glm-4.5v"

with open('config.yaml') as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
    vic_api_config = model_config.get("vic_api_config", None)
    model_config = model_config["model_config"]
# except:
#     raise ValueError("Model Config is not appropriate set")

set_vic_api_client(API_client(model_name,vic_api_config))
set_api_client(API_client(model_name,model_config))


class Config:
    dataset_file: str = "dataset/"  # 数据集文件路径
    image_path :str = "dataset/origin_images/"
    # 输出结果保存的根目录
    outcome_dir: str = "api_test/results_vic/"
    model_name: str = model_name
    # OpenRouter API Key配置
    openrouter_api_key: str = ""
    # 备用API Key（如需切换，取消注释下面的并注释上面的）
    # openrouter_api_key: str = ""

    # 请求限制配置
    rpm: int = 10  # 每分钟请求数限制（降低以避免429错误）
    request_delay: float = 0.5  # 每次请求之间的额外延时（秒），增加以减少重试
    round_delay: float = 0.1  # 每轮处理之间的延时（秒）
    max_no_improve_round_count: int = 20  # 最大连续未成功请求的轮数（增加以允许更多重试）
    process_count: int = 5  # 总共处理的数据项数量（用于测试）-1表示处理所有数据

    max_tokens: int = 8192  # API调用的最大token数（防止响应被截断）
    timeout: int = 120  # API请求超时时间（秒）
    max_retries: int = 5  # 每个API调用的最大重试次数
    retry_delay_base: float = 2.0  # 重试延迟基数（秒），将使用指数退避

    # OpenRouter客户端 - 将在main_async中初始化
    client: AsyncOpenAI = None


    @staticmethod
    def setup_logging(output_dir: str):
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


def create_output_dirs(config: Config) -> tuple:
    """创建输出目录并返回目录路径和详细日志文件路径"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name_part = config.model_name.split("/")[-1]
    output_dir = os.path.join(
        config.outcome_dir,
        config.dataset_file.split("/")[-1].split(".")[0],
        model_name_part,
        f"{timestamp}_process_count_{config.process_count}",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rounds_outcome"), exist_ok=True)

    # 创建详细日志文件路径
    detail_log_path = os.path.join(output_dir, "detailed_output.log")
    return output_dir, detail_log_path


# 这个函数用于将图片编码为base64（用于直接API调用）
def encode_image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"图片编码错误 {image_path}: {str(e)}")
        return ""


async def process_one_item(
        data: Dict, config: Config, limiter: aiolimiter.AsyncLimiter
) -> Dict:
    """负责处理单个数据项，调用 Agent Utils Pipeline"""

    # # [关键] 创建一个临时的 JSON 文件路径，确保并发时互不干扰
    # question_index = data.get('QUESTION_INDEX', 'unknown')
    # fd, temp_decision_path = tempfile.mkstemp(suffix=".json", prefix=f"decision_{question_index}_")
    # os.close(fd)  # 关闭文件描述符，后续由 Utils 代码自己打开

    async with limiter:
        # 添加请求前的延时
        await asyncio.sleep(config.request_delay)

        try:
            # 1. 准备图片路径
            image_full_path = os.path.join(config.image_path, data["QUESTION_INDEX"][:-2] + ".png")

            # 2. 准备 Query
            # [关键] 格式要求将在 execute_synthesis 阶段添加，这里只传递原始问题
            # 这样 decision_generation 阶段只会输出到 module's tasks，不会输出最终答案
            full_query = data["QUESTION"] + "\n Options:" + data["OPTION"]
            base64_image = [encode_image_to_base64(image_full_path)]

            # 5. 定义同步的 Agent 流程 (这就是你的 run_demo 逻辑)
            # def run_agent_pipeline():
            #     decision_generation(agent_args)
            #     split_task(agent_args)
            #     execute_modularization(agent_args)
            #     summary(agent_args)
            #     execute_synthesis(agent_args)
            #
            #     # 读取最终结果（包含所有中间步骤的输出）
            #     with open(temp_decision_path, 'r') as f:
            #         final_data = json.load(f)
            #     return final_data


            response = await Vic_async(query=full_query,image=base64_image)
            print(response)

            data["visual_inference_chain"] = str(response['visual_inference_chain'])
            data["intermediate_response"] = response['intermediate_response']
            data["raw_response"] = response['response']


            # 8. 获取最终响应并解析答案
            raw_response_content = response['response']
            # 9. 结果解析 (改进的解析逻辑，支持多种格式)
            answer = None

            # 优先尝试提取 XML 格式的 <ANSWER> 标签
            xml_pattern = r'<ANSWER>(.*?)</ANSWER>'
            xml_matches = re.findall(xml_pattern, raw_response_content, re.DOTALL | re.IGNORECASE)

            if xml_matches:
                answer = xml_matches[0].strip()

            # 如果XML格式未找到，尝试提取 "Answer: X" 格式（支持冒号后的多种变体）
            if not answer:
                answer_patterns = [
                    r'Answer:\s*([A-E])',  # Answer: A
                    r'Answer\s*:\s*([A-E])',  # Answer : A (允许空格)
                    r'answer:\s*([A-E])',  # 小写
                    r'Answer\s+([A-E])\s',  # Answer A (无冒号，但有空格)
                ]
                for pattern in answer_patterns:
                    matches = re.findall(pattern, raw_response_content, re.IGNORECASE)
                    if matches:
                        answer = matches[-1].strip().upper()  # 取最后一个匹配，转大写
                        break

            # 如果还是没找到，尝试从文本末尾提取单个字母 A-E
            if not answer:
                last_letter_match = re.search(r'\b([A-E])\b', raw_response_content[-200:], re.IGNORECASE)
                if last_letter_match:
                    answer = last_letter_match.group(1).upper()

            # 如果所有方法都失败，设置为 "No choice"
            if not answer:
                answer = "No choice"
                logging.warning(f"无法从响应中提取答案: {data.get('QUESTION_INDEX')}")

            data["is_success"] = True
            data["simple_answer"] = str(answer).strip()  # 确保 simple_answer 字段存在

            # 判断正误（不区分大小写）
            correct_answer = str(data.get("ANSWER", "")).strip().upper()
            predicted_answer = str(answer).strip().upper()
            data["final"] = (predicted_answer == correct_answer)

            # 记录详细信息（用于调试）
            if not data["final"]:
                logging.debug(f"问题 {data.get('QUESTION_INDEX')}: 预测={predicted_answer}, 正确答案={correct_answer}")

        except Exception as e:
            if response is not None:
                error_message = str(e) + "\t" + response.model_dump_json()
            else:
                error_message = str(e)
            logging.error(f"处理数据时出错: {error_message}")

            data["is_success"] =  False,
        # except Exception as e:
        #     import traceback
        #     import time
        #     error_message = f"Error in {data.get('QUESTION_INDEX', 'unknown')}: {str(e)}"
        #     error_traceback = traceback.format_exc()
        #     logging.error(error_message)
        #     logging.debug(f"详细错误信息: {error_traceback}")
        #
        #     # 根据错误类型决定是否值得重试
        #     error_str = str(e)
        #     is_retryable = False
        #     retry_delay = 0
        #
        #     if "Connection error" in error_str or "timeout" in error_str.lower():
        #         is_retryable = True
        #         retry_delay = 5.0  # 连接错误等待5秒
        #     elif "Expecting value" in error_str or "JSON" in error_str:
        #         # JSON解析错误可能是响应被截断，需要增加max_tokens后重试
        #         is_retryable = True
        #         retry_delay = 3.0  # JSON错误等待3秒
        #     elif "rate limit" in error_str.lower() or "429" in error_str:
        #         is_retryable = True
        #         retry_delay = 10.0  # 速率限制等待10秒
        #
        #     # 如果错误可重试，标记为失败但会在下一轮重试
        #     # 否则标记为最终失败
        #     data["is_success"] = False
        #     data["raw_response"] = error_message
        #     data["simple_answer"] = "Error"
        #     data["final"] = False
        #
        #     # 如果是可重试错误，记录信息以便下一轮处理
        #     if is_retryable:
        #         logging.info(f"任务 {data.get('QUESTION_INDEX')} 遇到可重试错误，将在下一轮重试")
        #         # 短暂延迟，避免立即重试导致的问题
        #         time.sleep(min(retry_delay, 2.0))

        # finally:
            # # [关键] 清理临时文件，防止磁盘塞满
            # if os.path.exists(temp_decision_path):
            #     os.remove(temp_decision_path)

        return data


# 下面的代码基本保持不变
async def process_round(
        config: Config,
        limiter: aiolimiter.AsyncLimiter,
        pending_items_map: Dict,
        final_results: Dict,
) -> Dict:
    tasks = []
    for index, item in pending_items_map.items():
        tasks.append(process_one_item(item, config, limiter))

    current_round_results = await tqdm_asyncio.gather(*tasks)

    for i, result in enumerate(current_round_results):
        index = list(pending_items_map.keys())[i]
        final_results[index] = result

    return final_results


async def main_async(mode):
    Mode = {
        "single": "single_qa_pair.json",
        "multi_trend": "multi_trend_qa_pair.json",
        "reason": "reason_qa_pair.json",
    }

    config = Config()

    output_dir, detail_log_path = create_output_dirs(config)
    config.setup_logging(output_dir)


    json_path = os.path.join(config.dataset_file, Mode[mode])
    round_outcome_dir = os.path.join(output_dir, "rounds_outcome")
    limiter = aiolimiter.AsyncLimiter(1, 60.0 / config.rpm)

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if config.process_count > 0:
        dataset = dataset[: config.process_count]

    final_results = copy.deepcopy(dataset)
    pending_items_map = {i: dataset[i] for i in range(len(dataset))}
    round_count = 0

    while (
            len(pending_items_map) > 0 and round_count < config.max_no_improve_round_count
    ):
        round_count += 1
        logging.info(f"第{round_count}轮，剩余未处理数据: {len(pending_items_map)}")

        final_results = await process_round(
            config, limiter, pending_items_map, final_results
        )
        with open(
                os.path.join(round_outcome_dir, f"round_{round_count}.json"),
                "w",
                encoding="utf-8",
        ) as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        # 失败重试逻辑
        pending_items_map = {
            i: final_results[i]
            for i in range(len(final_results))
            if not final_results[i]["is_success"]
        }
        if len(pending_items_map) == 0:
            break

        # 每轮处理之间添加延时，避免请求过于频繁
        if len(pending_items_map) > 0:
            logging.info(f"等待 {config.round_delay} 秒后开始下一轮处理...")
            await asyncio.sleep(config.round_delay)

    filename_base = mode + "_" + config.model_name.split("/")[-1] + "_vic_results"

    json_filename = filename_base + ".json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    logging.info(f"JSON结果已保存到: {json_path}")

    df = pd.DataFrame(final_results)
    csv_filename = filename_base + ".csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"CSV结果已保存到: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main_async("single"))