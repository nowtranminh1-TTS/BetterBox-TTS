# -*- mode: python ; coding: utf-8 -*-
"""
Viterbox TTS — PyInstaller onedir, ưu tiên Windows (CUDA + PyTorch wheel thường gặp).

  pyinstaller distributionApp.spec

Đầu ra: dist/ViterboxTTS/

Gom toàn bộ package có file .txt/.json/.npz trong site-packages (Gradio,
s3tokenizer, perth/resemble-perth, pedalboard, …). Nếu exe báo thiếu file
trong _internal/<tên_package>/, thêm tên import được vào _PACKAGES_COLLECT_ALL.

Dữ liệu trong repo (nếu có lúc build): viterbox/modelViterboxLocal, wavs, viterbox/pretrained, viterbox/output-profile,
downloads, general/config_path.txt — tự bundle; không copy build/ hay dist/ vào _internal khi phát hành.
"""
from __future__ import annotations

import pathlib
import sys

from PyInstaller.utils.hooks import collect_all

block_cipher = None

_spec_dir = pathlib.Path(SPEC).parent.resolve()
_app = str(_spec_dir / "app.py")

# Import name (pip có thể khác: resemble-perth → perth).
# collect_all = data + binaries + submodules — cần cho file tĩnh đọc lúc import.
_PACKAGES_COLLECT_ALL: tuple[str, ...] = (
    # --- Gradio 5 (UI) ---
    "gradio",
    "gradio_client",
    "safehttpx",
    "groovy",
    # --- Model phụ: tokenizer S3, CLAP/perth ---
    "s3tokenizer",
    "perth",
    # --- Hiệu ứng âm thanh (pedalboard) — nhiều file kèm package ---
    "pedalboard",
)


def _safe_collect_all(package: str) -> tuple[list, list, list]:
    try:
        return collect_all(package)
    except Exception as exc:
        print(f"[distributionApp.spec] WARN: bỏ qua collect_all({package!r}): {exc}", file=sys.stderr)
        return [], [], []


_BUNDLE_DATA_NAMES = (
    "viterbox/modelViterboxLocal",
    "wavs",
    "viterbox/pretrained",
    "viterbox/output-profile",
    "downloads",
)


def _append_project_datas(target: list, root: pathlib.Path) -> None:
    for name in _BUNDLE_DATA_NAMES:
        path = root / name
        if path.is_dir():
            target.append((str(path), name))
    cfg = root / "general" / "config_path.txt"
    if cfg.is_file():
        target.append((str(cfg), "."))


datas: list = []
binaries: list = []
hiddenimports: list = []

for _pkg in _PACKAGES_COLLECT_ALL:
    d, b, h = _safe_collect_all(_pkg)
    datas += d
    binaries += b
    hiddenimports += h

_append_project_datas(datas, _spec_dir)

hiddenimports += [
    "pydub",
    "pydub.utils",
    "soundfile",
]

a = Analysis(
    [_app],
    pathex=[str(_spec_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name="ViterboxTTS",
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ViterboxTTS",
)
