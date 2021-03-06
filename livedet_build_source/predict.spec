# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
site_packages = 'C:\\Users\\lab\\Documents\\Github\\Livedet2021\\livedet_build_source\\livedet38_env\\Lib\\site-packages'

a = Analysis(['predict.py'],
             pathex=['C:\\Users\\lab\\Documents\\Github\\Livedet2021\\livedet_build_source', 'C:\\Users\\lab\\Documents\\Github\\Livedet2021\\livedet_build_source\\livedet38_env\\Lib\\site-packages'],
             binaries=[],
             datas=[(os.path.join(site_packages, 'numpy'),'numpy'),
                    (os.path.join(site_packages, 'albumentations'),'albumentations'),
                    (os.path.join(site_packages, 'scipy'), 'scipy'),
                    (os.path.join(site_packages, 'skimage'), 'skimage'),
                    (os.path.join(site_packages, 'geffnet'), 'geffnet')],
             hiddenimports=['numpy.core.multiarray'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='hallymMMC',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
