conda activate base
Write-Output "Eliminando el environment si ya existia..."
conda env remove -n waste-detector
Write-Output "Enviro eliminado. Recreando..."
conda env create --file=win-config.yaml
conda activate waste-detector

conda install .\pycocotools-2.0.6-py310h9b08ddd_1.tar.bz2


pip install --extra-index-url https://download.pytorch.org/whl/cu116 torch==1.12.0+cu116 torchvision==0.13.0+cu116
pip install torchmetrics torchinfo
pip install codecarbon albumentations
pip install lightning
pip install importlib_metadata
pip install neptune-client neptune-contrib
pip install jupyter_contrib_nbextensions
pip install sphinx sphinx_rtd_theme
pip install pybboxes
pip install tensorboard
jupyter contrib nbextension install

Write-Output "Enviro recreado. Instalando ipykernel..."
python -m ipykernel install --user --name=waste-detector
Write-Output "ipykernel instalado."
Write-Output "Comprobando que todo se ha instalado correctamente..."
python -c "import torch, torchvision, lightning; print(torch.cuda.is_available())"
python -c "from waste_detection_system import *"
Write-Output "Todo hecho."