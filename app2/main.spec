# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],  # Your entry point
    pathex=['.'],  # Current directory
    binaries=[],
    datas=[
        ('*.py', '.'),  # Include all .py files in the current dir (except main.py as main.py is compiled)
        ('*.json', '.'),  # If you need package.json, params.json etc
	('C:/Users/youse/miniconda3/envs/fy_sim/Lib/site-packages/dash_cytoscape', 'dash_cytoscape'),
    ],
    hiddenimports=[
        'dash',
        'dash_bootstrap_components',
        'dash_cytoscape',
        'networkx',
        'math',
        'json',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='main'
)
