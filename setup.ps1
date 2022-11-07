conda activate base
conda env remove -n waste-detector
conda env create --file .\win-config.yaml
conda activate waste-detector
python -m ipykernel install --user --name=waste-detector
pip uninstall custom_torchvision custom_torchvision_reference_detection custom_ultralytics_yolov5
Set-Location custom_torchvision
python setup.py install
Set-Location ..
Set-Location custom_ultralytics_yolov5
python setup.py install
Set-Location ..
Set-Location custom_torchvision_reference_detection
python setup.py install
Set-Location ..
# python ./setup.py install