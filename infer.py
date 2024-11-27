import re

import backoff

from openai import OpenAI

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def infer(instruction: str, model_name: str):
    openai_api_key = OPENAI_API_KEY
    openai_api_base = OPENAI_API_BASE
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    temperature = 0.7
    max_tokens = 4096

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは優秀で誠実な日本人のアシスタントです。あなたはユーザーからの指示に対して忠実に従い、親切に回答します。"},
            {"role": "user", "content": instruction},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

@backoff.on_exception(backoff.fibo, Exception, max_tries=1000)
def evaluator(instruction: str, response1: str, response2: str, model_name: str, prompt: str) -> int:
    openai_api_key = OPENAI_API_KEY
    openai_api_base = OPENAI_API_BASE
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    temperature = 0.3
    max_tokens = 1024

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "あなたは与えられたAIアシスタントの応答のうち、どちらがユーザーにとって適切なものであるか判定するアシスタントです。"},
            {"role": "user", "content": prompt.replace("RESPONSE1", response1).replace("RESPONSE2", response2).replace("INSTRUCTION", instruction)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 評価結果を取得
    evaluation = int(response.choices[0].message.content.split("Winner: ")[1][0])

    # 1/0以外の値が返ってきた場合はエラー
    if evaluation not in [1, 2]:
        return -1

    return evaluation