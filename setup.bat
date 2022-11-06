conda activate base
conda deactivate
conda env remove -n waste-detector
conda env create --file .\win-config.yaml
conda activate waste-detector
python -m ipykernel install --user --name=waste-detector
pip uninstall custom_torchvision custom_torchvision_reference_detection custom_ultralytics_yolov5
cd custom_torchvision
python setup.py install
cd ..
cd custom_torchvision_reference_detection
python setup.py install
cd ..
cd custom_ultralytics_yolov5
python setup.py install
cd ..
python ./setup.py install