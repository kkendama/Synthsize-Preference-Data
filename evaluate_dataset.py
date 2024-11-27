from pathlib import Path
import argparse
import json
import collections

from datasets import Dataset, load_dataset, concatenate_datasets

from infer import evaluator

def get_evaluation_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt

def run_evaluate(args):
    # output_dirが存在しない場合は作成
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    prompt = get_evaluation_prompt(args.prompt_path)

    # datasetのmodelごとの最大のindexを取得し、その中で最小のindexを取得
    for model in list(set(dataset["model"])):
        model_dataset = dataset.filter(lambda x: x["model"] == model)
        model_index = max(model_dataset["index"])
        last_index = min(last_index, model_index)

    for i in range(last_index + 1):
        index_dataset = dataset.filter(lambda x: x["index"] == i)
        instruction = index_dataset["instruction"][0]
        responses = index_dataset["output"]
        # 0からlen(responses)-1までの組み合わせをDatasetDictとして作成
        pairs = [{"response1": i, "response2": j} for i in range(len(responses)) for j in range(len(responses))]
        pairs_dataset = Dataset.from_list(pairs)
        pairs_dataset = pairs_dataset.filter(lambda x: x["response1"] == x["response2"])
        pairs_dataset = concatenate_datasets([pairs_dataset, pairs_dataset])

        def evaluate_responses(x):
            response1 = responses[x["response1"]]
            response2 = responses[x["response2"]]
            response = evaluator(instruction, response1, response2, args.model_name, prompt)
            if response == 1:
                winner = x["response1"]
            else:
                winner = x["response2"]
            return winner
        
        pairs_dataset = pairs_dataset.map(lambda x: {"winner": evaluate_responses(x)}, num_proc=args.batch_size)
        winner_count = collections.Counter(pairs_dataset["winner"])
        ranking = sorted(winner_count.items(), key=lambda x: -x[1])
        # 票数とresposneをセットにする
        ranking = [(x[0], x[1], responses[x[0]]) for x in ranking]
        # 1位と最下位の票数が異なる場合にはデータを保存
        if ranking[0][1] != ranking[-1][1]:
            output = {
                "index": i,
                "conversations": [{"role": "user", "content": instruction}],
                "chosen": responses[ranking[0][0]],
                "rejected": responses[ranking[-1][0]],
                "ranking": ranking
            }
            with open(Path(args.output_dir, "evaluation.jsonl"), "a") as f:
                f.write(json.dumps(output) + "\n")
        else:
            output = {
                "index": i,
                "conversations": [{"role": "user", "content": instruction}],
                "ranking": ranking
            }
            with open(Path(args.output_dir, "no_winner.jsonl"), "a") as f:
                f.write(json.dumps(output) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument("--data_path", type=str, default="output/responses.json", help="Path to dataset")
    parser.add_argument("--prompt_path", type=str, default="evaluation_prompt.txt", help="Path to evaluation prompt")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    args = parser.parse_args()

    run_evaluate(args)