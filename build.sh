rm -rf output
mkdir output
python3 setup.py bdist_wheel
mv dist/*.whl output/
pip install output/*.whl
