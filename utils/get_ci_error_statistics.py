import argparse
import json
import math
import os
import subprocess
import time
import zipfile
from collections import Counter

import requests


def get_job_links(workflow_run_id):
    run_id = workflow_run_id
    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{run_id}/jobs?per_page=100"
    result = requests.get(url).json()
    jobs = {}

    try:

        print(result.keys())

        if "documentation_url" in result:
            print(result["documentation_url"])
        if "message" in result:
            print(result["message"])

        if "jobs" in result:
            _jobs = result["jobs"]
            for _job in _jobs:
                if "name" in _job:
                    print(_job["name"])
                else:
                    print("None")
                print("-" * 40)

        print("=" * 80)

        jobs.update({job["name"]: job["html_url"] for job in result["jobs"]})
        pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)
        print(pages_to_iterate_over)

        for i in range(pages_to_iterate_over):
            print(i)
            result = requests.get(url + f"&page={i + 2}").json()

            print(result.keys())

            if "documentation_url" in result:
                print(result["documentation_url"])
            if "message" in result:
                print(result["message"])

            if "jobs" in result:
                _jobs = result["jobs"]
                for _job in _jobs:
                    if "name" in _job:
                        print(_job["name"])
                    else:
                        print("None")
                    print("-" * 40)
            print("=" * 80)

            jobs.update({job["name"]: job["html_url"] for job in result["jobs"]})

        print(len(jobs))
        print("Extract warnings in CI artifacts" in jobs)

        return jobs
    except Exception as e:
        print(i)
        print("Unknown error, could not fetch links.", e)

    return {}


def get_artifacts_links(worflow_run_id):
    """Get all artifact links from a workflow run"""

    url = f"https://api.github.com/repos/huggingface/transformers/actions/runs/{worflow_run_id}/artifacts?per_page=100"
    result = requests.get(url).json()
    artifacts = {}

    try:
        artifacts.update({artifact["name"]: artifact["archive_download_url"] for artifact in result["artifacts"]})
        pages_to_iterate_over = math.ceil((result["total_count"] - 100) / 100)

        for i in range(pages_to_iterate_over):
            result = requests.get(url + f"&page={i + 2}").json()
            artifacts.update({artifact["name"]: artifact["archive_download_url"] for artifact in result["artifacts"]})

        return artifacts
    except Exception as e:
        print("Unknown error, could not fetch links.", e)

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--workflow_run_id", default=None, type=str, required=True, help="A GitHub Actions workflow run id."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Where to store the downloaded artifacts and other result files.",
    )
    parser.add_argument(
        "--token", default=None, type=str, required=True, help="A token that has actions:read permission."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    artifacts_links = get_artifacts_links(args.worflow_run_id)

    _job_links = get_job_links(args.workflow_run_id)
    job_links = {}
    # To deal with `workflow_call` event, where a job name is the combination of the job names in the caller and callee.
    # For example, `PyTorch 1.11 / Model tests (models/albert, single-gpu)`.
    if _job_links:
        for k, v in _job_links.items():
            # This is how GitHub actions combine job names.
            if " / " in k:
                index = k.find(" / ")
                k = k[index + len(" / ") :]
            job_links[k] = v
    with open(os.path.join(args.output_dir, "job_links.json"), "w", encoding="UTF-8") as fp:
        json.dump(job_links, fp, ensure_ascii=False, indent=4)

    print(job_links)
