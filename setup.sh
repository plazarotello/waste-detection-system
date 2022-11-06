#!/bin/bash
conda activate base
conda deactivate
conda env remove -n waste-detector
conda env create --file ./unix-config.yaml
conda activate waste-detector
python -m ipykernel install --user --name=waste-detector
pip uninstall custom_torchvision
python ./setup.py