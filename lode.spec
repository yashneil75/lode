# -*- mode: python ; coding: utf-8 -*-
"""One-file `lode` binary.

fastembed pulls in onnxruntime (native DLLs) and the Rust `tokenizers` extension, and
both ship data files that the static analysis misses, so everything is collected whole.
The embedding model itself is NOT bundled -- it is downloaded to LODE_HOME on first use,
which keeps the binary small and lets users pick a different model.
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for package in ("fastembed", "onnxruntime", "tokenizers", "numpy", "rich", "pathspec"):
    pkg_data, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_data
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["packaging/entry.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["lode.cli"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "transformers", "sentence_transformers", "matplotlib", "pandas", "scipy", "pytest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lode",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# One dir, not one file: an onefile binary unpacks itself to a temp folder on every
# run, which costs ~3 s before `lode list` even starts. A directory of libraries boots
# in a third of that, and the release is a single archive either way.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="lode",
)
