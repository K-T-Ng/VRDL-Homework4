# VRDL-Homework4

## Requirements
The following packages are used in this homework
```
torch==1.9.0+cu111
torchvision==0.10.0+cu111
Pillow==8.1.0
tqdm==4.55.1
```

## Folder structure

    .
    ├──datasets
       ├──testing_lr_images     # produced in next section
          ├──XXXX.png
       ├──training_hr_images    # produced in next section
          ├──XXXX.png
       ├──validation_hr_images  # produced in next section
          ├──XXXX.png
       ├──TrHR                  # produced in next section
          ├──2x
            ├──XXXX.png
          ├──3x
            ├──XXXX.png
          ├──4x
            ├──XXXX.png
       ├──TrLR                  # produced in next section
       ├──ValHR                 # produced in next section
       ├──ValLR                 # produced in next section
       ├──Valid                 # produced during training
    ├──answer
       ├──XXXX_pred.png
    ├──function
    ├──model
    ├──weights
    ├──inference.py
    ├──prepare_train_dataset.py
    ├──prepare_valid_dataset.py
    ├──train.py
    ├──train_multi_scale.py
    └──README.md
    
After cloning this repository,  the folder structure may looks like above, excepts those folder with comment ```# ...``` <br />

## Prepare Dataset
```Step1```: Divide your Training HR images into two parts, and put them into ```training_hr_images``` and ```validation_hr_images```, respectively. <br />
```Step2```: Put your testing LR images into ```testing_lr_images``` <br />
```Step3```: In order to produce ```TrHR```, ```TrLR```, ```ValHR``` and ```ValLR```, run
```
python prepare_train_dataset.py
python prepare_valid_dataset.py
```

## Training code
If you want to run single-scale training, modify the hyper-parameters in ```train.py``` (e.g. you can change the scaling up factor in ```line 28```). Then run 
```
python train.py
```
If you want to run multi-scale training, modify the hyper-parameters in ```train_multi_scale.py``` (e.g. you may increase the number of epochs in ```line 40```). Then run
```
python train_multi_scale.py
```

## Pre-trained model
After training, the top five weight file ```{ep}_{valid_psnr}.pth``` can be found in ```weights``` folder.
For reproducing the result, you may download the pretrained weight in https://drive.google.com/file/d/1Vd9YVL-zkJgEOm8nWHag5QBspLPFpC1E/view?usp=sharing. <br />
This weight file named ```1474_31.5458.pth```. After downloading this file, put it into ```weights``` folder.

## Inference code
You may change the weight folder in ```line 53``` in ```inference.py``` and run
```
python inference.py
```

## Reproduce the result
If you want to reproduce the result only, here we provide a few steps: <br />
```Step1```: Clone this repo. <br />
```Step2```: Put testing LR images into ```datasets/testing_lr_images``` <br />
```Step3```: Download pre-trained weight file from https://drive.google.com/file/d/1Vd9YVL-zkJgEOm8nWHag5QBspLPFpC1E/view?usp=sharing. <br />
```Step4```: Put the pre-trained weight file ```1474_31.5458.pth``` to ```weights``` folder. <br />
```Step5```: Change ```line 53``` in ```inference.py``` into ```ckpt = os.path.join('weights', '1474_31.5458.pth')```. <br />
```Step6```: Run ```python inference.py```. <br />
```Step7```: You may get the resulting HR images in ```answer``` folder.

## Reference
We use the code from https://github.com/saeed-anwar/DRLN
