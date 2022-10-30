## Run the code base
### Configure environments
Create virtual python environment
```sh
conda create -n harmony python=3.7
conda activate harmony
```
Install the required python packages.
```sh
pip install -r  requirements.txt
```

### Run and compare harmony methods 
Run ComBat: 
```sh
python main_demo.py  -harmony_mode=ComBat  -feature_name=Demo --harmony_retrain=1
```
Run CycleGAN:
```sh
python main_demo.py  -harmony_mode=ComBat  -feature_name=Demo --harmony_retrain=1
```
Run MCD-GAN:
```sh
python main_demo.py  -harmony_mode=MCDGAN  -feature_name=Demo  --harmony_retrain=1 --lambda_discrepancy_control=3.2
```
Visulizing results
```sh
python demo_visualize.py
```
![Methods Comparison](./code/result/Picture1.png)
