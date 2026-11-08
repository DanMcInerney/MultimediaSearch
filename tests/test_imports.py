"""The heavy libraries stay out of mms's own process (DESIGN.md §13): only `worker/` (the model's own process) imports torch,
transformers, vllm or pplx_decider, and importing the deciders, `gpu`, `modelenv`, the transport, `fetch`, `prepare` and the
adapter loads none of them, nor huggingface_hub, which only `fetch.py` imports, when a download runs."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from conftest import ROOT

HEAVY = ("torch", "transformers", "vllm", "pplx_decider")
HUB = "huggingface_hub"
SOURCE = ROOT / "src" / "mms"
IMPORTED = ("mms.decider", "mms.decider_d1", "mms.decider_d1_3b", "mms.decider_d3", "mms.decider_pplx",
            "mms.decider_h2o", "mms.gpu", "mms.modelenv", "mms.transport", "mms.fetch",
            "mms.prepare", "mms.adapter")


def heavy_imports(path: Path, libraries: tuple = HEAVY) -> list[str]:
    """Each `import X` or `from X import ...` in the file, at any depth (a function's included), whose X is one of
    `libraries` (default: the heavy ones)."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        found += [f"{path.name}:{node.lineno} {name}" for name in names if name.split(".")[0] in libraries]
    return found


def outside_the_worker() -> list[Path]:
    return [path for path in sorted(SOURCE.rglob("*.py")) if "worker" not in path.relative_to(SOURCE).parts[:-1]]


def test_no_module_outside_the_worker_imports_a_heavy_library_anywhere():
    outside = outside_the_worker()
    assert len(outside) > 20 and SOURCE / "prepare.py" in outside and SOURCE / "worker" / "d1.py" not in outside
    assert [line for path in outside for line in heavy_imports(path)] == []


def test_fetch_is_the_one_module_outside_the_worker_that_imports_the_hub():
    importers = {path.name for path in outside_the_worker() if heavy_imports(path, (HUB,))}
    assert importers == {"fetch.py"}


def test_the_scan_sees_an_import_in_a_function_and_a_from_import(tmp_path):
    one = tmp_path / "x.py"
    one.write_text("def f():\n    import torch\n    from transformers.models import auto\n    import os\n    import torchish\n",
                   encoding="utf-8")
    assert [line.split(" ")[1] for line in heavy_imports(one)] == ["torch", "transformers.models"]
    hub = tmp_path / "y.py"
    hub.write_text("def f():\n    import huggingface_hub\n    from huggingface_hub import constants\n", encoding="utf-8")
    assert [line.split(" ")[1] for line in heavy_imports(hub, (HUB,))] == [HUB, HUB]
    assert heavy_imports(hub) == [], "the hub is not one of the heavy libraries"


def test_importing_the_deciders_gpu_modelenv_the_transport_fetch_prepare_and_the_adapter_loads_no_torch_or_hub():
    code = ("import " + ", ".join(IMPORTED) + "; import sys; "
            f"print([name for name in {(*HEAVY, HUB)} if name in sys.modules] or False)")
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", done.stdout
