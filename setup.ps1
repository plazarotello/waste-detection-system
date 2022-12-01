conda activate base
Write-Output "Eliminando el environment si ya existia..."
conda env remove -n waste-detector
Write-Output "Enviro eliminado. Recreando..."
conda create --name waste-detector -y python=3.10
conda activate waste-detector

#conda config --env --add channels pytorch 
#conda config --env --add channels conda-forge
#conda config --env --add channels anaconda

conda install --force-reinstall -y -c anaconda openssl
conda install --force-reinstall -y conda 
conda install --force-reinstall -y pillow==9.2.0 
conda install --force-reinstall -y numpy 
conda install --force-reinstall -y pandas 
conda install --force-reinstall -y opencv 
conda install --force-reinstall -y matplotlib 
conda install --force-reinstall -y libpng 
conda install --force-reinstall -y jpeg 
conda install --force-reinstall -y ffmpeg 
conda install --force-reinstall -y ipykernel 
conda install --force-reinstall -y ipywidgets 
conda install --force-reinstall -y ipython 
conda install --force-reinstall -y cython 
conda install --force-reinstall -y notebook 
conda install --force-reinstall -y nb_conda_kernels 
conda install --force-reinstall -y nbconvert 
conda install --force-reinstall -y yaml 
conda install --force-reinstall -y scikit-learn 
conda install --force-reinstall -y scipy 
#conda install --force-reinstall -y seaborn 
conda install --force-reinstall -y tqdm

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchmetrics
pip install torchinfo
pip install codecarbon
pip install ninja
conda install .\pycocotools-2.0.6-py310h9b08ddd_1.tar.bz2
Write-Output "Enviro recreado. Instalando ipykernel..."
python -m ipykernel install --user --name=waste-detector
Write-Output "ipykernel instalado. Eliminando paquetes custom..."
pip uninstall -y torchvision custom_torchvision_reference_detection custom_ultralytics_yolov5
Write-Output "Instalando custom_torchvision..."
Set-Location custom_torchvision
Remove-Item -Recurse -Force -ErrorAction Ignore dist
Remove-Item -Recurse -Force -ErrorAction Ignore build
Remove-Item -Recurse -Force -ErrorAction Ignore torchvision.egg-info
python setup.py install *>&1 > installation.log
Remove-Item -Recurse -Force -ErrorAction Ignore dist
Remove-Item -Recurse -Force -ErrorAction Ignore build
Remove-Item -Recurse -Force -ErrorAction Ignore torchvision.egg-info
Set-Location ..
Write-Output "Instalando custom_ultralytics_yolov5..."
Set-Location custom_ultralytics_yolov5
Remove-Item -Recurse -Force -ErrorAction Ignore dist
Remove-Item -Recurse -Force -ErrorAction Ignore build
Remove-Item -Recurse -Force -ErrorAction Ignore custom_ultralytics_yolov5.egg-info
python setup.py install *>&1 > installation.log
Remove-Item -Recurse -Force -ErrorAction Ignore dist
Remove-Item -Recurse -Force -ErrorAction Ignore build
Remove-Item -Recurse -Force -ErrorAction Ignore custom_ultralytics_yolov5.egg-info
Set-Location ..
Write-Output "Instalando custom_torchvision_reference_detection..."
Set-Location custom_torchvision_reference_detection
Remove-Item -Recurse -ErrorAction Ignore -Force dist
Remove-Item -Recurse -ErrorAction Ignore -Force build
Remove-Item -Recurse -ErrorAction Ignore -Force custom_torchvision_reference_detection.egg-info
python setup.py install *>&1 > installation.log
Set-Location custom_torchvision_reference_detection
Remove-Item -Recurse -ErrorAction Ignore -Force dist
Remove-Item -Recurse -ErrorAction Ignore -Force build
Remove-Item -Recurse -ErrorAction Ignore -Force custom_torchvision_reference_detection.egg-info
Set-Location ..
Write-Output "Comprobando que los paquetes custom estan bien instalados..."
python -c "import torchvision"
python -c "import custom_torchvision_reference_detection"
python -c "import custom_ultralytics_yolov5"
Write-Output "Comprobando que todo se ha instalado correctamente..."
python -c "from waste_detection_system import *"
Write-Output "Todo hecho."
# python ./setup.py install