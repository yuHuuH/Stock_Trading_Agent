
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DIST_APP = BASE / "dist" / "TradingAgentDemo"


def run(cmd: list[str], check: bool = True) -> int:
    print("\n>", " ".join(str(x) for x in cmd))
    completed = subprocess.run(cmd, cwd=BASE)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed.returncode


def command_available(command: str) -> bool:
    try:
        completed = subprocess.run(
            [command, "--version"],
            cwd=BASE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return completed.returncode == 0
    except FileNotFoundError:
        return False


def pip_available() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        cwd=BASE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def install_requirements(requirements_file: str) -> None:
    req_path = BASE / requirements_file
    if not req_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {req_path}")

    # uv-first because the user's venv may not include pip.
    if command_available("uv"):
        run(["uv", "pip", "install", "--python", sys.executable, "-r", requirements_file])
        return

    # Fallback for normal Python environments.
    if pip_available():
        run([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        return

    # Last attempt: bootstrap pip if available.
    print("\nuv was not found and pip is not available.")
    print("Trying to bootstrap pip with ensurepip...")
    run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False)

    if pip_available():
        run([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        return

    raise RuntimeError(
        "Could not install requirements because neither uv nor pip is available.\n"
        "Since you are using uv, make sure the `uv` command is available in this terminal.\n"
        "Try: uv --version"
    )


def ensure_runtime_files() -> None:
    required_options = [
        (BASE / "data" / "SP500.csv", "data/SP500.csv"),
        (BASE / "data" / "sp500_2025_20260.csv", "data/sp500_2025_20260.csv"),
        (BASE / "artifacts" / "vec_normalize_stats.pkl", "artifacts/vec_normalize_stats.pkl"),
    ]

    missing = []
    for path, label in required_options:
        if not path.exists():
            missing.append(label)

    model_ok = (
        (BASE / "artifacts" / "best_model.zip").exists()
        or (BASE / "artifacts" / "ppo_quant_bot_final.zip").exists()
    )
    if not model_ok:
        missing.append("artifacts/best_model.zip OR artifacts/ppo_quant_bot_final.zip")

    if missing:
        print("\nWARNING: Some runtime files are missing:")
        for item in missing:
            print(f"  - {item}")
        print("\nThe EXE can still be built, but the app will ask for these files at runtime.")


def copy_runtime_folders() -> None:
    (DIST_APP / "data").mkdir(parents=True, exist_ok=True)
    (DIST_APP / "artifacts").mkdir(parents=True, exist_ok=True)

    if (BASE / "data").exists():
        shutil.copytree(BASE / "data", DIST_APP / "data", dirs_exist_ok=True)

    if (BASE / "artifacts").exists():
        shutil.copytree(BASE / "artifacts", DIST_APP / "artifacts", dirs_exist_ok=True)

    if (BASE / ".streamlit").exists():
        shutil.copytree(BASE / ".streamlit", DIST_APP / ".streamlit", dirs_exist_ok=True)


def main() -> None:
    print("Building TradingAgentDemo with PyInstaller.")
    print(f"Python used: {sys.executable}")

    ensure_runtime_files()

    install_requirements("requirements.txt")
    install_requirements("requirements-build.txt")

    shutil.rmtree(BASE / "build", ignore_errors=True)
    shutil.rmtree(BASE / "dist", ignore_errors=True)

    run([sys.executable, "-m", "PyInstaller", "TradingAgentDemo.spec", "--clean", "--noconfirm"])

    copy_runtime_folders()

    print("\nBuild complete.")
    print(f"Run: {DIST_APP / 'TradingAgentDemo.exe'}")
    print("Distribute the whole dist/TradingAgentDemo folder, not just the .exe.")


if __name__ == "__main__":
    main()
