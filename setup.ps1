conda activate base
Write-Output "Eliminando el environment si ya existia..."
conda env remove -n waste-detector
Write-Output "Enviro eliminado. Recreando..."
conda env create --file .\win-config.yaml
conda activate waste-detector
Write-Output "Enviro recreado. Instalando ipykernel..."
python -m ipykernel install --user --name=waste-detector
Write-Output "ipykernel instalado. Eliminando paquetes custom..."
pip uninstall -y torchvision custom_torchvision_reference_detection custom_ultralytics_yolov5
Write-Output "Instalando custom_torchvision..."
Set-Location custom_torchvision
Remove-Item -Recurse -Force dist
Remove-Item -Recurse -Force build
Remove-Item -Recurse -Force torchvision.egg-info
set DISTUTILS_USE_SDK=1
python setup.py install *>&1 > installation.log
Set-Location ..
Write-Output "Instalando custom_ultralytics_yolov5..."
Set-Location custom_ultralytics_yolov5
Remove-Item -Recurse -Force dist
Remove-Item -Recurse -Force build
Remove-Item -Recurse -Force custom_ultralytics_yolov5.egg-info
python setup.py install *>&1 > installation.log
Set-Location ..
Write-Output "Instalando custom_torchvision_reference_detection..."
Set-Location custom_torchvision_reference_detection
Remove-Item -Recurse -Force dist
Remove-Item -Recurse -Force build
Remove-Item -Recurse -Force custom_torchvision_reference_detection.egg-info
python setup.py install *>&1 > installation.log
Set-Location ..
Write-Output "Comprobando que los paquetes custom estan bien instalados..."
python -c "import torchvision"
python -c "import custom_torchvision_reference_detection"
python -c "import custom_ultralytics_yolov5"
Write-Output "Comprobando que todo se ha instalado correctamente..."
python -c "import waste_detection_system"
Write-Output "Todo hecho."
# python ./setup.py install