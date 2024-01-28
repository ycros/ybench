import argparse
import os
from typing import Dict, Optional
import requests
import subprocess
from dataclasses import dataclass


@dataclass
class Metadata:
    base_url: str
    ctx_length: int
    model_name: str
    prog_name: str
    prog_ver: str
    commit_id: Optional[str] = None
    dir_name: Optional[str] = None


def get_metadata_from_api(url: str, key: str) -> Optional[str]:
    try:
        response = requests.get(url).json()
        return response.get(key)
    except:
        return None


def get_metadata(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    metadata = {}

    metadata['base_url'] = args.base_url
    metadata['model_name'] = args.model_name or get_metadata_from_api(f"{args.base_url}/api/v1/model", "result")
    metadata['ctx_length'] = args.ctx_length or get_metadata_from_api(f"{args.base_url}/api/extra/true_max_context_length", "value")
    metadata['prog_name'] = args.prog_name or get_metadata_from_api(f"{args.base_url}/api/extra/version", "result")
    metadata['prog_ver'] = args.prog_ver or get_metadata_from_api(f"{args.base_url}/api/extra/version", "version")

    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
        dir_name = os.path.basename(os.getcwd())
        metadata["commit_id"] = commit_id
        metadata["dir_name"] = dir_name
    except:
        pass

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="API Benchmarking Tool")
    parser.add_argument("--base-url", default="http://localhost:5001/", help="Base URL of the API")
    parser.add_argument("--ctx-length", type=int, default=None, help="Max total tokens limit for benchmarking")
    parser.add_argument("--model-name", default=None, help="Model name if API call fails")
    parser.add_argument("--prog-name", default=None, help="Program name if API call fails")
    parser.add_argument("--prog-ver", default=None, help="Program version if API call fails")
    return parser.parse_args(), parser


def validate_metadata(metadata: Dict[str, Optional[str]]) -> Optional[Metadata]:
    found_error = False
    for key in metadata:
        if metadata[key] is None and key not in ["commit_id", "dir_name"]:
            found_error = True
            print(f"Metadata '{key}' is missing. Consider passing it via a command-line argument.")

    if not found_error:
        return Metadata(**metadata) # type: ignore - checked above


if __name__ == "__main__":
    args, parser = parse_args()
    raw_metadata = get_metadata(args)
    metadata = validate_metadata(raw_metadata)
    if metadata is None:
        parser.print_help()
    print(metadata)