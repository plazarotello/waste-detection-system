eval "$(conda shell.bash hook)"
conda activate base
echo "Eliminando el environment si ya existia..."
conda env remove -n waste-detector
rm -rf /anaconda/envs/waste-detector
echo "Enviro eliminado. Recreando..."
conda env create --file unix-config.yaml
conda activate waste-detector
echo "Enviro recreado. Instalando ipykernel..."
python -m ipykernel install --user --name=waste-detector
echo "ipykernel instalado. Eliminando paquetes custom..."
conda remove -y torchvision
pip install -y pyyaml
pip uninstall -y torchvision custom_torchvision_reference_detection custom_ultralytics_yolov5
echo "Instalando custom_torchvision..."
cd custom_torchvision
python setup.py install > installation.log 2>&1
cd ..
echo "Instalando custom_ultralytics_yolov5..."
cd custom_ultralytics_yolov5
python setup.py install > installation.log 2>&1
cd ..
echo "Instalando custom_torchvision_reference_detection..."
cd custom_torchvision_reference_detection
python setup.py install > installation.log 2>&1
cd ..
python -c "import torchvision"
python -c "import custom_torchvision_reference_detection"
python -c "import custom_ultralytics_yolov5"
echo "Todo hecho."
# python ./setup.py install
