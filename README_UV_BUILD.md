
# UV build instructions

This version is patched for `uv` environments where `.venv` may not include `pip`.

## Build with uv

From the project folder:

```bash
build_exe.bat
```

or:

```bash
build_exe_uv.bat
```

or:

```bash
.venv\Scripts\python.exe build_exe.py
```

The scripts use:

```bash
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -r requirements-build.txt
```

So they do **not** require `python -m pip`.

## Manual uv commands

```bash
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe -r requirements-build.txt
.venv\Scripts\python.exe -m PyInstaller TradingAgentDemo.spec --clean --noconfirm
```

After building:

```text
dist\TradingAgentDemo\TradingAgentDemo.exe
```

Distribute the whole folder:

```text
dist\TradingAgentDemo\
```


## v8 Streamlit PyInstaller fix

This version fixes the bundled EXE error:

```text
RuntimeError: server.port does not work when global.developmentMode is true.
```

The launcher now forces Streamlit production mode with:

```text
--global.developmentMode=false
```

It also includes `.streamlit/config.toml` with `developmentMode = false`.
