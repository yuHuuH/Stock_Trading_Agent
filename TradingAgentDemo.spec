
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

block_cipher = None

datas = [
    ("app.py", "."),
    ("README.md", "."),
    (".streamlit/config.toml", ".streamlit"),
]

hiddenimports = []

# Streamlit needs package metadata and static frontend assets.
for package in [
    "streamlit",
    "altair",
    "plotly",
    "pandas",
    "numpy",
    "gymnasium",
    "gym_anytrading",
    "stable_baselines3",
    "torch",
    "sklearn",
]:
    try:
        pkg_datas, pkg_bins, pkg_hidden = collect_all(package)
        datas += pkg_datas
        hiddenimports += pkg_hidden
    except Exception:
        pass

# Some packages are imported dynamically.
for package in [
    "streamlit.web",
    "streamlit.runtime",
    "plotly.validators",
    "gym_anytrading.envs",
    "stable_baselines3.common",
    "torch.distributions",
]:
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

# Metadata helps importlib.metadata lookups used by Streamlit and dependencies.
for package in [
    "streamlit",
    "altair",
    "plotly",
    "pandas",
    "numpy",
    "gymnasium",
    "gym-anytrading",
    "stable-baselines3",
    "torch",
    "scikit-learn",
]:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

a = Analysis(
    ["trading_agent_launcher.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "wandb",
        "kaggle",
        "kaggle_secrets",
        "pandas_ta",
        "numba",
        "llvmlite",
        "tensorflow",
        "matplotlib.tests",
        "numpy.tests",
        "pandas.tests",
        "torch.testing",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TradingAgentDemo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TradingAgentDemo",
)
