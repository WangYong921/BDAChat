import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset

from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import get_model_name_from_path
from videollava.utils import disable_torch_init

from videollava.eval.inference import run_inference
from videollava.eval.classification import classification_metrics
from videollava.eval.detection import detection_metrics

DATASET_TO_JSON = {
	"olbda_qa": "olbda_qa.json",
}
def load_model(model_path, model_base, load_8bit=False, load_4bit=False, cache_dir=None, device=None):

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, _ = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_4bit=load_4bit,
        load_8bit=load_8bit,
        device=device,
        cache_dir=cache_dir,
    )

    model.model.video_tower = None

    processor = processor['image']
    return tokenizer, model, processor
def load_local_dataset(data_root, dataset_name):
    json_file = DATASET_TO_JSON.get(dataset_name)
    if not json_file:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    json_path = os.path.join(data_root, json_file)
    print(f"[DEBUG]json:{json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    data_dir = os.path.dirname(json_path)
    for example in dataset:
        if 'video' in example:
            example['video'] = [os.path.join(data_dir, img) for img in example['video']]
            for img_path in example['video']:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
    return dataset

def eval(
        dataset_name,
        model_path,
        model_base,
        load_8bit=False, 
        load_4bit=False,
        cache_dir=None,
        local_data_dir=None,
        out_name=None,
        out_dir=None,
        prompt_strategy=None, 
        chronological_prefix=True,
        conv_mode='v1',
        device='cuda',
        force_rerun=False,
        temperature=0.2,
        max_new_tokens=256
):
    args = locals()
    print(f"Arguments passed to eval:")
    for k, v in args.items():
        print(f"\t{k} ({type(v).__name__}): {v}")

    detection_datasets = [
        "olbda_qa",
    ]

    elif dataset_name in detection_datasets:
        eval_metrics_fn = detection_metrics
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_name2hf_split = {
        "olbda_qa": "olbda_qa",
    }
    hf_split = dataset_name2hf_split[dataset_name]

    if out_dir is None:
        out_dir = Path("results")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    out_subdir = out_dir / dataset_name
    out_subdir.mkdir(exist_ok=True)

    if out_name is None:
        model_name = get_model_name_from_path(model_path)
        out_name = f"{model_name}.json"

    if ".json" not in out_name:
        out_name = f"{out_name}.json"

    args_to_determine_path = [
        'prompt_strategy',
        'chronological_prefix',
    ]
    for arg in args_to_determine_path:
        if args[arg] is not None:
            out_name = out_name.replace(".json", f"_{arg}_{args[arg]}.json")

    out_path = out_subdir / out_name

    if out_path.exists() and not force_rerun:
        print(f"Output file {out_path} already exists. Computing metrics without running inference.")
        with open(out_path, "r") as f:
            outputs = json.load(f)

    else:
        tokenizer, model, processor = load_model(
            model_path,
            model_base,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            cache_dir=cache_dir,
            device=device,
        )

        dataset = load_local_dataset(local_data_dir, dataset_name)
        outputs = run_inference(
            dataset,
            model,
            tokenizer,
            processor,
            prompt_strategy,
            chronological_prefix,
            conv_mode,
            temperature,
            max_new_tokens
        )
        print(f"Saving outputs to {out_path}")
        with open(out_path, "w") as f:
            json.dump(outputs, f, indent=4)

    metrics = eval_metrics_fn(outputs, dataset_name=dataset_name)
    print(f"Metrics for dataset {dataset_name}:")
    for key, value in metrics.items():
        print(f"\t{key}: {value}")


def str_or_none(value):
    if value == "" or value.lower() == "none":
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str_or_none, default=None, required=False)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_data_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--prompt_strategy", type=str, default="interleave")
    parser.add_argument("--chronological_prefix", action="store_true") 
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    eval(**vars(args))
