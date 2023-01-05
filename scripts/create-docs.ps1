conda activate waste-detector
cd docs
make clean
sphinx-apidoc -o .\source\ ..\waste_detection_system\
cd ..
docs/make html