"""Execute the walkthrough in this Python environment and save inline outputs."""

import json
import os
import sys
from pathlib import Path

import yaml


def main() -> None:
    """Use a project-local kernel specification; no global kernel registration."""
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "configs/walkthrough.yaml").read_text(encoding="utf-8"))
    runtime = root / "artifacts/notebooks/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for variable, folder in {
        "JUPYTER_RUNTIME_DIR": "jupyter", "IPYTHONDIR": "ipython",
        "JUPYTER_CONFIG_DIR": "config", "MPLCONFIGDIR": "matplotlib",
    }.items():
        path = runtime / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)

    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    kernel_root = runtime / "kernels"
    kernel_dir = kernel_root / "self-healing-machines"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Self-Healing Machines (.venv)", "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(
        kernel_name="self-healing-machines",
        connection_file=str(runtime / "jupyter/kernel-walkthrough.json"),
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )
    notebook_path = root / "notebooks/walkthrough.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            if index == 0 or notebook.cells[index-1].cell_type != "markdown":
                raise ValueError(f"Code cell {index} needs a preceding Markdown explanation.")

    def show_progress(cell: dict, cell_index: int, **kwargs: object) -> None:
        """Report completed cells so execution is easy to follow."""
        if cell["cell_type"] == "code":
            print(f"Executed code cell {cell['execution_count']}", flush=True)

    client = NotebookClient(
        notebook, km=manager, timeout=settings["cell_timeout_seconds"],
        resources={"metadata": {"path": str(root / "notebooks")}},
        on_cell_executed=show_progress,
    )
    print(f"Starting notebook kernel with {sys.executable}", flush=True)
    try:
        client.execute()
    finally:
        if client.kc is not None:
            client.kc.stop_channels()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    print(f"Saved {notebook_path}: {len(code_cells)} executed code cells.")


if __name__ == "__main__":
    main()
