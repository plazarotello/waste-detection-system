conda deactivate
conda env remove -n waste-detector
conda activate base
conda env create --file .\config.yaml
conda activate waste-detector
python -m ipykernel install --name=waste-detector
python .\setup.py