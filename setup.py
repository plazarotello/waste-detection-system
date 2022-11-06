import os
import site
import platform
from pathlib import Path
import shutil
import subprocess
import sys

if __name__=='__main__':
    if platform.system() == 'Windows':
        package_to_find = 'torchvision'
        package_to_update = Path(__file__).parent.resolve()/'src'\
            /'custom_torchvision'
        suffixes_to_copy = ('.dll', '.pyd')
        paths_to_check = site.getsitepackages()

        matches = []
        for path in paths_to_check:
            for dirpath, _, filenames in os.walk(Path(path)):
                    if dirpath.endswith(package_to_find):
                        matches = matches + [os.path.join(dirpath, f) 
                            for f in filenames if 
                            f.endswith(suffixes_to_copy)]
        #copy
        print(f'Matches: {matches}')
        print(f'Copy to: {package_to_update}')
        for _match in matches:
            shutil.copy(src=str(_match), dst=str(package_to_update), 
                        follow_symlinks=True)

    elif platform.system() == 'Linux':
        custom_build = Path(__file__).parent.resolve()\
            /'custom_torchvision_build_0.13.0'
        build_src = custom_build / 'custom_torchvision'
        updated_src = Path(__file__).parent.resolve()/'src'\
            /'custom_torchvision'
        
        shutil.rmtree(build_src)
        os.makedirs(build_src, exist_ok=True)
        only_pys = lambda dir, files : [f for f in files if \
            os.path.isfile(os.path.join(dir, f)) and f.endswith('.py')]
        shutil.copytree(src=updated_src, dst=build_src, ignore=only_pys)

        # build the package
        exec_file = custom_build/'setup.py'
        subprocess.call([sys.executable, str(exec_file), 'install'])
        
    else: raise 'No computer version'