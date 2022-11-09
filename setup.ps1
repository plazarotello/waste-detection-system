conda activate base
Write-Output "Eliminando el environment si ya existia..."
conda env remove -n waste-detector
Write-Output "Enviro eliminado. Recreando..."
conda env create --file .\win-config.yaml
conda activate waste-detector
Write-Output "Enviro recreado. Instalando ipykernel..."
python -m ipykernel install --user --name=waste-detector
Write-Output "ipykernel instalado. Eliminando paquetes custom..."
conda remove -y torchvision
pip uninstall -y torchvision custom_torchvision_reference_detection custom_ultralytics_yolov5
Write-Output "Instalando custom_torchvision..."
Set-Location custom_torchvision
set DISTUTILS_USE_SDK=1
python setup.py install *>&1 > installation.log
Set-Location ..
Write-Output "Instalando custom_ultralytics_yolov5..."
Set-Location custom_ultralytics_yolov5
python setup.py install *>&1 > installation.log
Set-Location ..
Write-Output "Instalando custom_torchvision_reference_detection..."
Set-Location custom_torchvision_reference_detection
python setup.py install *>&1 > installation.log
Set-Location ..
python -c "import torchvision"
python -c "import custom_torchvision_reference_detection"
python -c "import custom_ultralytics_yolov5"
Write-Output "Todo hecho."
# python ./setup.py install