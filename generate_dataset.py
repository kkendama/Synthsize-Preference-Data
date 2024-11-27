from pathlib import Path
import argparse
import json

from datasets import DatasetDict, Dataset, load_dataset

from infer import infer

def get_evaluation_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt

def get_dataset(dataset_name: str, split: str) -> DatasetDict:
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def run_generate_response(args, dataset: DatasetDict) -> None:
    # output_dirが存在しない場合は作成
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # 前回処理したデータがある場合は読み込む
    if Path(args.output_dir, "responses.jsonl").exists():
        generated_dataset = load_dataset("json", data_files=str(Path(args.output_dir, "responses.jsonl")), split="train")
        generated_dataset = generated_dataset.filter(lambda x: x["model"] == args.model_name)
        last_index = max(generated_dataset["index"])
    else:
        last_index = -1

    # 今回処理するデータの最終indexを取得
    end_index = min(last_index + args.sample_size, len(dataset))
    dataset = dataset.select(range(last_index + 1, end_index))
    print(f"Start generating responses from index {last_index} to {end_index}")

    index = last_index + 1

    for record in dataset:
        instruction = record[args.column_name]
        subset = Dataset.from_list([{"instruction": instruction} for _ in range(args.num_generations)])
        subset = subset.map(lambda x: {"output": infer(x["instruction"], args.model_name)}, num_proc=args.batch_size)
        # 結果を保存
        with open(Path(args.output_dir, "responses.jsonl"), "a") as f:
            for data in subset:
                output = {"index": index, "model": args.model_name, "instruction": instruction, "output": data["output"]}
                f.write(json.dumps(output) + "\n")
        index += 1

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for evaluation")
    parser.add_argument("--dataset_name", type=str, default="conv_ai", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--column_name", type=str, default="context", help="Column name")
    parser.add_argument("--prompt_path", type=str, default="evaluation_prompt.txt", help="Path to evaluation prompt")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of generations")
    args = parser.parse_args()

    dataset = get_dataset(args.dataset_name, args.split)
    print(f"Loaded dataset {args.dataset_name} {args.split}")

    run_generate_response(args, dataset)

if __name__ == "__main__":
    main()