__version__ = '0.13.0a0+6b60f92'
git_version = '6b60f92502ae0c58df55caf95a0ec56234629d4a'
from custom_torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
