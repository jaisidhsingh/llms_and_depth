import uuid
import os
import argparse
import subprocess


def make_submisison_file_content(executable, arguments, output, error, log, cpus=1, gpus=0, memory=1000, disk="1G"):
    d = {
        'executable': executable,
        'arguments': arguments,
        'output': output,
        'error': error,
        'log': log,
        'request_cpus': cpus,
        'request_gpus': gpus,
        'request_memory': memory,
        'request_disk': disk,
        # 'requirements': '(TARGET.Machine != "g154") && (TARGET.Machine != "g164")'
    }
    return d

def run_job(uid, bid, d):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    job_file = os.path.join('tmp', uid)
    with open(job_file, 'w') as f:
        for key, value in d.items():
            f.write(f'{key} = {value}\n')
        f.write("queue")

    subprocess.run(["condor_submit_bid", str(bid), job_file])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bid", type=int, default=25)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--memory", type=str, default="2G")
    parser.add_argument("--disk", type=str, default="2G")
    args = parser.parse_args()

    PROJECT_ROOT = "/home/jsingh/projects/llms_and_depth"

    uid = uuid.uuid4().hex[:10]
    arguments = ""
    config = {}
    for k, v in config.items():
        arguments += f"{v} "

    runs_folder = os.path.join(PROJECT_ROOT, "runs")
    os.makedirs(runs_folder, exist_ok=True)

    output = f"{runs_folder}/{uid}.stdout"
    error = f"{runs_folder}/{uid}.stderr"
    log = f"{runs_folder}/{uid}.log"
    executable = os.path.join(
        PROJECT_ROOT,
        "run_experiments",
        "import_test.sh"
    )

    try:
        content = make_submisison_file_content(
            executable,
            arguments,
            output,
            error,
            log,
            args.cpus,
            args.gpus,
            args.memory,
            args.disk
        )
        run_job(uid, args.bid, content)
    except:
        raise ValueError("Crashed.")
    print("Done.")
