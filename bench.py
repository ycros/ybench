import argparse
from curses import meta
from math import floor
import os
from typing import Dict, List, Optional, Tuple
from flask import request
import requests
import subprocess
from dataclasses import dataclass
import time
import datasets


TOKENS_TO_STRLEN_FACTOR = 4.27


@dataclass
class Metadata:
    base_url: str
    ctx_length: int
    model_name: str
    prog_name: str
    prog_ver: str
    commit_id: Optional[str] = None
    dir_name: Optional[str] = None


_dataset: Optional[datasets.Dataset] = None


def get_data(seed: int, length: int) -> str:
    global _dataset
    if _dataset is None:
        print("Downloading dataset...")
        dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("Dataset is not a HuggingFace Dataset")
        _dataset = dataset
        print("Dataset downloaded")
    shuffled = _dataset.shuffle(seed)
    data = ""
    for i in range(len(shuffled)):
        data += shuffled[i]["text"]
        if len(data) >= length:
            return data[:length]
    raise ValueError("Dataset is too small")


def tokenize(metadata: Metadata, prompt: str) -> List[int]:
    # POST request to /extra/tokencount with {"prompt": prompt}
    # Response is {"ids": [int]}
    data = {"prompt": prompt}
    response = requests.post(
        f"{metadata.base_url}/api/extra/tokencount", json=data
    ).json()
    return response["ids"]


def get_max_prompt(metadata: Metadata, seed: int) -> Tuple[str, int]:
    # We need a prompt greater than metadata.ctx_length, we use tokenize to get the length in tokens
    # get_data returns a string, but ctx_length is in tokens
    data_length = floor(metadata.ctx_length * TOKENS_TO_STRLEN_FACTOR)
    while True:
        prompt = get_data(seed, data_length)
        tokens = tokenize(metadata, prompt)
        # print("tokens", len(tokens))
        if len(tokens) > metadata.ctx_length:
            break
        data_length = floor(1.1 * data_length)
    # search backwards until we're within ctx_length/32 tokens of the limit
    token_chunk = metadata.ctx_length / 32
    max_ctx_limit = metadata.ctx_length - token_chunk
    # print("chunk", token_chunk)
    # print("max_ctx_limit", max_ctx_limit)
    # print("prompt", len(prompt))
    while True:
        prompt = prompt[: -floor(token_chunk * TOKENS_TO_STRLEN_FACTOR)]
        # print("prompt", len(prompt))
        tokens = tokenize(metadata, prompt)
        # print("tokens", len(tokens))
        if len(tokens) <= max_ctx_limit:
            break
    return prompt, len(tokens)


def benchmark_requests(metadata: Metadata):
    results = []
    seed = 0
    ctx_size_chunk = floor(metadata.ctx_length / 8)
    for iteration in range(1):
        for max_tokens in range(
            ctx_size_chunk, metadata.ctx_length - ctx_size_chunk, ctx_size_chunk
        ):
            prompt_length = 0
            while True:
                seed += 1

                max_prompt, max_prompt_token_count = get_max_prompt(metadata, seed)

                prompt_length += floor(ctx_size_chunk * TOKENS_TO_STRLEN_FACTOR)
                prompt = max_prompt[:prompt_length]
                prompt_tokens = tokenize(metadata, prompt)

                if len(prompt_tokens) + max_tokens > metadata.ctx_length:
                    break

                try:
                    result = generate(
                        metadata,
                        seed,
                        prompt,
                        max_tokens,
                    )
                    # result["prompt_length"] = prompt_length
                    result["prompt_tokens"] = len(prompt_tokens)
                    result["gen_tokens"] = max_tokens
                    print(result)
                    results.append(result)
                except requests.RequestException as e:
                    # log failure with extra details
                    print(e)
                    print(f"Failed with prompt length {prompt_length}")
                    print(f"Prompt: {prompt}")
                    print(f"Max tokens: {max_tokens}")
                    print(f"Seed: {seed}")
                    print(f"Metadata: {metadata}")
                    raise e

    return results


def generate(metadata: Metadata, seed: int, prompt: str, max_tokens: int) -> dict:
    request_data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "ignore_eos": True,
        "top_p": 1.0,
        "top_k": 0,
        "rep_pen": 1.0,
        "sampler_seed": seed,
    }

    start_time = time.perf_counter()
    response = requests.post(f"{metadata.base_url}/v1/completions", json=request_data)
    response.raise_for_status()
    duration = time.perf_counter() - start_time
    # response_data = response.json()
    # print(response_data)
    return {
        "duration": duration,
        # "tokens_generated": response_data["usage"]["completion_tokens"],
    }


def get_metadata_from_api(url: str, key: str) -> Optional[str]:
    try:
        response = requests.get(url).json()
        return response.get(key)
    except:
        return None


def get_metadata(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    metadata = {}

    metadata["base_url"] = args.base_url
    metadata["model_name"] = args.model_name or get_metadata_from_api(
        f"{args.base_url}/api/v1/model", "result"
    )
    metadata["ctx_length"] = args.ctx_length or get_metadata_from_api(
        f"{args.base_url}/api/extra/true_max_context_length", "value"
    )
    metadata["prog_name"] = args.prog_name or get_metadata_from_api(
        f"{args.base_url}/api/extra/version", "result"
    )
    metadata["prog_ver"] = args.prog_ver or get_metadata_from_api(
        f"{args.base_url}/api/extra/version", "version"
    )

    try:
        commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode()
        )
        dir_name = os.path.basename(os.getcwd())
        metadata["commit_id"] = commit_id
        metadata["dir_name"] = dir_name
    except:
        pass

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="API Benchmarking Tool")
    parser.add_argument(
        "--base-url", default="http://localhost:5001/", help="Base URL of the API"
    )
    parser.add_argument(
        "--ctx-length",
        type=int,
        default=None,
        help="Max total tokens limit for benchmarking",
    )
    parser.add_argument(
        "--model-name", default=None, help="Model name if API call fails"
    )
    parser.add_argument(
        "--prog-name", default=None, help="Program name if API call fails"
    )
    parser.add_argument(
        "--prog-ver", default=None, help="Program version if API call fails"
    )
    return parser.parse_args(), parser


def validate_metadata(metadata: Dict[str, Optional[str]]) -> Optional[Metadata]:
    found_error = False
    for key in metadata:
        if metadata[key] is None and key not in ["commit_id", "dir_name"]:
            found_error = True
            print(
                f"Metadata '{key}' is missing. Consider passing it via a command-line argument."
            )

    if not found_error:
        return Metadata(**metadata)  # type: ignore - checked above


if __name__ == "__main__":
    args, parser = parse_args()
    raw_metadata = get_metadata(args)
    metadata = validate_metadata(raw_metadata)
    if metadata is None:
        parser.print_help()
        exit(1)
    print(metadata)
    # get_max_prompt(metadata, 0)
    results = benchmark_requests(metadata)
    # print(results)
