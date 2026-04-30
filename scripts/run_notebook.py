"""execute a notebook in place using jupyter_client

minimal stand-in for `jupyter nbconvert --execute --inplace` since the venv
does not have nbconvert and the project's preferred installer (uv) is not
on path on this machine

streams cell sources and any stdout/stderr/error tracebacks to the terminal
so a tee'd log captures progress overwrites the input notebook with
executed outputs at the end (only after all cells succeed)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_output
from jupyter_client.manager import start_new_kernel


def _drain_iopub(kc, parent_msg_id: str, cell_outputs: list, *, timeout: float = 60 * 60):
    """pull iopub messages tied to parent_msg_id until the kernel reports idle

    appends nbformat-shaped output dicts to cell_outputs as they arrive and
    streams stdout/stderr text to the terminal so tail -f sees progress
    raises on cell errors so the runner can stop
    """
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except Exception as e:
            raise TimeoutError(f"iopub timeout waiting for kernel reply ({e})")
        if msg["parent_header"].get("msg_id") != parent_msg_id:
            continue
        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "status" and content["execution_state"] == "idle":
            return
        if msg_type == "stream":
            text = content.get("text", "")
            stream_name = content.get("name", "stdout")
            (sys.stdout if stream_name == "stdout" else sys.stderr).write(text)
            (sys.stdout if stream_name == "stdout" else sys.stderr).flush()
            cell_outputs.append({"output_type": "stream", "name": stream_name, "text": text})
        elif msg_type in {"execute_result", "display_data"}:
            cell_outputs.append(
                {
                    "output_type": msg_type,
                    "data": content.get("data", {}),
                    "metadata": content.get("metadata", {}),
                    **({"execution_count": content["execution_count"]} if msg_type == "execute_result" else {}),
                }
            )
        elif msg_type == "error":
            tb = "\n".join(content.get("traceback", []))
            print(f"\n[cell error] {content.get('ename')}: {content.get('evalue')}\n{tb}", flush=True)
            cell_outputs.append(
                {
                    "output_type": "error",
                    "ename": content.get("ename", ""),
                    "evalue": content.get("evalue", ""),
                    "traceback": content.get("traceback", []),
                }
            )
            raise RuntimeError(f"cell raised {content.get('ename')}: {content.get('evalue')}")


def run(notebook_path: Path, *, kernel_name: str = "python3", timeout: float = 60 * 60) -> None:
    nb = nbformat.read(str(notebook_path), as_version=4)
    print(f"[run_notebook] starting kernel {kernel_name}", flush=True)
    km, kc = start_new_kernel(kernel_name=kernel_name, cwd=str(notebook_path.parent))
    started = time.time()
    try:
        execution_count = 0
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            execution_count += 1
            elapsed = time.time() - started
            preview = (cell.source[:80] + "...") if len(cell.source) > 80 else cell.source
            preview = preview.replace("\n", " ")
            print(f"\n[cell {i + 1}/{len(nb.cells)} #{execution_count} t={elapsed:.0f}s] {preview}", flush=True)
            cell_outputs: list = []
            msg_id = kc.execute(cell.source)
            _drain_iopub(kc, msg_id, cell_outputs, timeout=timeout)
            cell["outputs"] = cell_outputs
            cell["execution_count"] = execution_count
        print(f"\n[run_notebook] all cells executed in {time.time() - started:.0f}s", flush=True)
        nbformat.write(nb, str(notebook_path))
        print(f"[run_notebook] wrote {notebook_path}", flush=True)
    finally:
        km.shutdown_kernel(now=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--timeout", type=float, default=60 * 60, help="per-cell iopub timeout in seconds")
    args = ap.parse_args()
    run(Path(args.notebook).resolve(), kernel_name=args.kernel, timeout=args.timeout)


if __name__ == "__main__":
    main()
