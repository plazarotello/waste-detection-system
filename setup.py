from setuptools import setup

with open("README.md") as f:
    readme = f.read()
with open("LICENSE") as f:
    license = f.read()

setup(
   name='waste-detector-paper-plastic',
   version='0.1',
   packages=['waste_detection_system'],
   long_description=readme,
   license=license,
   install_requires=[
    'pytorch==0.12.0'
    'torchvision==0.13.*', 
    'custom_torchvision_reference_detection', 
    'custom_ultralytics_yolov5'],
)