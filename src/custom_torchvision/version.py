__version__ = '0.13.0'
git_version = 'da3794e90c7cf69348f5446471926729c55f243e'
from custom_torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
