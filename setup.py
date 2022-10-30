from setuptools import setup, find_packages
import src as msw_detector
from pip._internal.req import parse_requirements
from pathlib import Path

current_directory = Path(__file__).parent.resolve()
install_reqs = parse_requirements(current_directory/'requirements.txt', session=False)
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name='msw-detector',
    version=msw_detector.__version__,
    description='Municipal Solid Waste Detector over a conveyor belt',
    author='Patricia Lázaro Tello',
    author_email='patricia.lazarotello@gmail.com',
    license=str(current_directory/'LICENSE'),
    long_description=open(str(current_directory/'README.md')).read(),
    install_requires=reqs,
    keywords=['municipal-solid-waste', 'waste-sorting', 'computer-vision', 
              'deep-learning', 'object-detection', 'image-recognition',
              'pytorch', 'yolov5', 'fasterrcnn', 'fcos', 'ssd', 'retinanet'],
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=['Environment :: Console',
                 'Environment :: GPU :: NVIDIA CUDA',
                 'Development Status :: 2 - Pre-Alpha',
                 'License :: OSI Approved :: GNU General Public License (GPL)',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Image Recognition'],
    packages = find_packages()
)