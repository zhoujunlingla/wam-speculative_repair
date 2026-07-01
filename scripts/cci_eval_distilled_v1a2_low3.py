#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path('/mnt/afs/intern/manlichen/ivan/zhoujunl')
CODE = ROOT / 'Wam_Speed_up' / 'lingbot-va-specverify'
ROBOTWIN_ROOT = ROOT / 'Wam_Speed_up' / 'RoboTwin'
PYTHON_PKGS = ROOT / 'env' / 'python_pkgs'
TORCH29_CLEAN_PKGS = ROOT / 'env' / 'torch29_clean_pkgs'
CUROBO_V1 = ROOT / 'env' / 'curobo-v0.7.8' / 'src'
NINJA_BIN = ROOT / 'env' / 'ninja-build' / 'extract' / 'usr' / 'bin'
TORCH_EXTENSIONS = ROOT / 'env' / 'torch_extensions'
CUDA_RUNTIME_LIB = Path('/usr/local/cuda-12.1/targets/x86_64-linux/lib')
NVIDIA_550_ROOT = ROOT / 'env' / 'nvidia-550.90.07-jammy' / 'extract'
NVIDIA_550_LIB = NVIDIA_550_ROOT / 'usr' / 'lib' / 'x86_64-linux-gnu'
NVIDIA_550_ICD = NVIDIA_550_ROOT / 'usr' / 'share' / 'vulkan' / 'icd.d' / 'nvidia_icd.json'

TEACHER_CLEAN_RATE = {
    'hanging_mug': 0.40,
    'turn_switch': 0.44,
    'place_can_basket': 0.81,
    'open_microwave': 0.82,
    'press_stapler': 0.85,
    'put_object_cabinet': 0.85,
    'stack_bowls_three': 0.86,
    'put_bottles_dustbin': 0.87,
    'dump_bin_bigbin': 0.89,
    'pick_diverse_bottles': 0.89,
}
DEFAULT_TASKS = ['hanging_mug', 'turn_switch', 'place_can_basket']

def ensure_dirs(*paths: Path) -> None:
    for root in paths:
        for name in ['logs', 'results', 'server']:
            (root / name).mkdir(parents=True, exist_ok=True)

def base_env(gpu: int, *, use_torch29: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['ROBOTWIN_ROOT'] = str(ROBOTWIN_ROOT)
    python_paths = []
    if use_torch29:
        python_paths.append(str(TORCH29_CLEAN_PKGS))
    python_paths.extend([str(PYTHON_PKGS), str(CUROBO_V1)])
    python_paths.append(env.get('PYTHONPATH', ''))
    env['PYTHONPATH'] = ':'.join(p for p in python_paths if p)
    env['PATH'] = f'{NINJA_BIN}:/usr/local/cuda/bin:{env.get("PATH", "")}'
    env['TORCH_EXTENSIONS_DIR'] = str(TORCH_EXTENSIONS)
    env['LIBRARY_PATH'] = f'{CUDA_RUNTIME_LIB}:{env.get("LIBRARY_PATH", "")}'
    env['LD_LIBRARY_PATH'] = f'{NVIDIA_550_LIB}:{CUDA_RUNTIME_LIB}:{env.get("LD_LIBRARY_PATH", "")}'
    env['VK_ICD_FILENAMES'] = str(NVIDIA_550_ICD)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['DIFFUSERS_DISABLE_BITSANDBYTES'] = '1'
    env['PYTHONWARNINGS'] = 'ignore::UserWarning'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    env['PYTHONUNBUFFERED'] = '1'
    return env

def start_server(run_root: Path, *, gpu: int, port: int, master_port: int, model_path: Path, video_steps: int, action_steps: int) -> subprocess.Popen:
    log_path = run_root / 'logs' / f'server_g{gpu}.log'
    log = log_path.open('a', buffering=1)
    env = base_env(gpu, use_torch29=True)
    env['EVAL_MODEL_PATH'] = str(model_path)
    env['EVAL_VIDEO_STEPS'] = str(video_steps)
    env['EVAL_ACTION_STEPS'] = str(action_steps)
    cmd = [
        sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', '1',
        '--master_port', str(master_port), 'wan_va/wan_va_server.py',
        '--config-name', 'robotwin_eval_ckpt', '--save_root', str(run_root / 'server' / 'distilled'),
        '--port', str(port),
    ]
    print(f'[eval] start server gpu={gpu} port={port} model={model_path}', flush=True)
    return subprocess.Popen(cmd, cwd=str(CODE), env=env, stdout=log, stderr=subprocess.STDOUT)

def read_metric(root: Path, task: str) -> dict[str, object]:
    p = root / 'results' / 'stseed-10000' / 'metrics' / task / 'res.json'
    if not p.exists():
        return {'task': task, 'succ_num': 0, 'total_num': 0, 'succ_rate': 0.0, 'teacher_clean_rate': TEACHER_CLEAN_RATE.get(task, 0.0), 'delta_vs_teacher': -TEACHER_CLEAN_RATE.get(task, 0.0), 'status': 'missing'}
    data = json.loads(p.read_text())
    total = int(data.get('total_num', 0)); succ = int(data.get('succ_num', 0))
    rate = float(data.get('succ_rate', succ / total if total else 0.0))
    ref = TEACHER_CLEAN_RATE.get(task, 0.0)
    return {'task': task, 'succ_num': succ, 'total_num': total, 'succ_rate': rate, 'teacher_clean_rate': ref, 'delta_vs_teacher': rate - ref, 'status': 'ok'}

def summarize(run_root: Path, result_root: Path, *, tasks: list[str], start_time: str, end_time: str | None, content: str, model_path: Path, video_steps: int, action_steps: int, test_num: int) -> None:
    rows = [read_metric(run_root, t) for t in tasks]
    succ = sum(int(r['succ_num']) for r in rows); total = sum(int(r['total_num']) for r in rows)
    rate = succ / total if total else 0.0
    summary = {
        'run_root': str(run_root), 'result_root': str(result_root), 'start_time': start_time,
        'end_time': end_time, 'experiment_content': content, 'model_path': str(model_path),
        'config_name': 'robotwin_eval_ckpt', 'video_steps': video_steps, 'action_steps': action_steps,
        'tasks': tasks, 'test_num_per_task': test_num, 'success': succ, 'total': total,
        'success_rate': rate, 'updated_at': datetime.now().isoformat(timespec='seconds'), 'rows': rows,
    }
    for root in (run_root, result_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / 'summary_latest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (root / 'per_task_success.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['task','succ_num','total_num','succ_rate','teacher_clean_rate','delta_vs_teacher','status'])
            writer.writeheader(); writer.writerows(rows)
        lines = [f'# Distilled v1/a2 Step500 Small Eval', '', f'- Start: {start_time}', f'- End: {end_time or "RUNNING"}', f'- Experiment: {content}', f'- Model: `{model_path}`', f'- Steps: video={video_steps}, action={action_steps}', f'- Current: {succ}/{total} = {rate*100:.2f}%', '', '| Task | Success | Total | Rate | Ref | Delta | Status |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for r in rows:
            lines.append(f"| {r['task']} | {r['succ_num']} | {r['total_num']} | {float(r['succ_rate'])*100:.2f}% | {float(r['teacher_clean_rate'])*100:.2f}% | {float(r['delta_vs_teacher'])*100:+.2f} pp | {r['status']} |")
        (root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run_task(run_root: Path, result_root: Path, *, task: str, client_gpu: int, port: int, test_num: int) -> tuple[int, str]:
    log_path = run_root / 'logs' / f'client_{task}.log'
    env = base_env(client_gpu, use_torch29=False)
    cmd = [sys.executable, '-m', 'evaluation.robotwin.eval_polict_client_openpi', '--config', 'policy/ACT/deploy_policy.yml', '--host', '127.0.0.1', '--port', str(port), '--overrides', '--task_name', task, '--task_config', 'demo_clean', '--train_config_name', '0', '--model_name', '0', '--ckpt_setting', 'distilled-v1a2-step500', '--seed', '0', '--policy_name', 'ACT', '--save_root', str(run_root / 'results'), '--video_guidance_scale', '5', '--action_guidance_scale', '1', '--test_num', str(test_num)]
    with log_path.open('a', buffering=1) as log:
        log.write(f'[task] {task} tn={test_num} gpu={client_gpu}\n')
        log.write('[command] ' + ' '.join(cmd) + '\n')
        proc = subprocess.run(cmd, cwd=str(CODE), env=env, stdout=log, stderr=subprocess.STDOUT)
    (result_root / 'logs' / f'client_{task}.log').write_text(log_path.read_text(errors='replace'), encoding='utf-8')
    metric = read_metric(run_root, task)
    if proc.returncode != 0 and int(metric['total_num']) >= test_num:
        return 0, f'POST_RESULT_RC_{proc.returncode}'
    return proc.returncode, 'ok' if proc.returncode == 0 else f'RC_{proc.returncode}'

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', required=True); p.add_argument('--result-root', required=True)
    p.add_argument('--model-path', required=True); p.add_argument('--server-gpu', type=int, required=True)
    p.add_argument('--client-gpu', type=int, required=True); p.add_argument('--port', type=int, required=True)
    p.add_argument('--master-port', type=int, required=True); p.add_argument('--server-warmup-sec', type=int, default=240)
    p.add_argument('--test-num', type=int, default=3); p.add_argument('--video-steps', type=int, default=1); p.add_argument('--action-steps', type=int, default=2)
    p.add_argument('--label', default='distilled-v1a2-step500'); p.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    args = p.parse_args()
    tasks = args.tasks
    run_root = Path(args.run_root); result_root = Path(args.result_root); model_path = Path(args.model_path)
    ensure_dirs(run_root, result_root)
    content = f'{args.label}: small RobotWin clean eval on {len(tasks)} low-success tasks, TN={args.test_num}'
    start = datetime.now().isoformat(timespec='seconds')
    for root in (run_root, result_root):
        root.mkdir(parents=True, exist_ok=True)
        (root / 'command.sh').write_text(' '.join(sys.argv) + '\n', encoding='utf-8')
        (root / 'start_time.txt').write_text(start + '\n', encoding='utf-8')
        (root / 'experiment_content.txt').write_text(content + '\n', encoding='utf-8')
        (root / 'task_list.json').write_text(json.dumps(tasks, indent=2), encoding='utf-8')
    summarize(run_root, result_root, tasks=tasks, start_time=start, end_time=None, content=content, model_path=model_path, video_steps=args.video_steps, action_steps=args.action_steps, test_num=args.test_num)
    server = None
    try:
        server = start_server(run_root, gpu=args.server_gpu, port=args.port, master_port=args.master_port, model_path=model_path, video_steps=args.video_steps, action_steps=args.action_steps)
        print(f'[eval] warming server for {args.server_warmup_sec}s', flush=True)
        time.sleep(args.server_warmup_sec)
        for i, task in enumerate(tasks, 1):
            print(f'[eval] task {i}/{len(tasks)} {task}', flush=True)
            rc, status = run_task(run_root, result_root, task=task, client_gpu=args.client_gpu, port=args.port, test_num=args.test_num)
            print(f'[eval] task {task} status={status}', flush=True)
            summarize(run_root, result_root, tasks=tasks, start_time=start, end_time=None, content=content, model_path=model_path, video_steps=args.video_steps, action_steps=args.action_steps, test_num=args.test_num)
            if rc != 0:
                raise RuntimeError(f'task {task} failed with {status}')
        summarize(run_root, result_root, tasks=tasks, start_time=start, end_time=datetime.now().isoformat(timespec='seconds'), content=content, model_path=model_path, video_steps=args.video_steps, action_steps=args.action_steps, test_num=args.test_num)
        print('[eval] complete', flush=True)
    finally:
        if server and server.poll() is None:
            server.send_signal(signal.SIGTERM); time.sleep(5)
            if server.poll() is None:
                server.kill()

if __name__ == '__main__':
    main()
