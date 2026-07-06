"""Minimum smoke tests: the package must compile and import."""
import subprocess
import sys
from pathlib import Path


def test_compileall():
    pkg = Path(__file__).parent.parent / "daph_exfusion"
    result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(pkg)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"compileall failed:\n{result.stdout}\n{result.stderr}"


def test_import_merge_toolkit():
    from daph_exfusion.merge_toolkit import MemoryBankExFusionFFN, SwiGLUFFN
    assert SwiGLUFFN is not None


def test_import_optional_mlx_does_not_crash():
    import daph_exfusion
    assert daph_exfusion.__version__ == "2026.07.4.3.5"
