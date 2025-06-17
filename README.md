# JOSENet: A Joint Stream Embedding Network for Violence Detection in Surveillance Videos
Josenet: Official Pytorch implementation for "JOSENet: A Joint Stream Embedding Network for Violence Detection in Surveillance Videos" *(paper under review)*


## Fine-tune & Inference
### 1. Env Setting
```
conda env create -f environment.yml
conda activate josenet
```

### 2. Prepare Data
- HMDB51: run `python tools/hmdb51_download.py` <- This first
- RWF-2000: [Box Drive](https://duke.app.box.com/s/kfgnl5bfy7w75cngopskms8kbh5w1mvu), unzip and run `python tools/rwf_2000_dataset_creation.py` <- This second

### 3. Auxiliary SSL train
```
python auxiliary.py --model_name model_ssl
```

### 4. Fine-Tuning
You must prepare your dataset.
If you don't have violence video dataset, just use our model when you inference.
(You can check our fine-tuning result in results folder.)
```
python fine-tune.py
```

## Abstract 📖
Due to the ever-increasing availability of video surveillance cameras and the growing need for crime prevention, the violence detection task is attracting greater attention from the research community. With respect to other action recognition tasks, violence detection in surveillance videos shows additional issues, such as the presence of a significant variety of real fight scenes. Unfortunately, available datasets seem to be very small compared with other action recognition datasets. Moreover, in surveillance applications, people in the scenes always differ for each video and the background of the footage differs for each camera. Also, violent actions in real-life surveillance videos must be detected quickly to prevent unwanted consequences, thus models would definitely benefit from a reduction in memory usage and computational costs. Such problems make classical action recognition methods difficult to be adopted. To tackle all these issues, we introduce JOSENet, a novel self-supervised framework that provides outstanding performance for violence detection in surveillance videos. The proposed model receives two spatiotemporal video streams, i.e., RGB frames and optical flows, and involves a new regularized self-supervised learning approach for videos. JOSENet provides improved performance while requiring one-fourth of the number of frames per video segment and a reduced frame rate compared to state-of-the-art methods.

## Installation requirements ⚙️
The code is based on python 3.10.6. All the modules can be installed using: `pip install -r requirements.txt`.

Another possibility is to directly install the Anaconda environment: 
- `conda env create -f environment.yml`
- `conda activate josenet`


## Training 📉
Hyperparameters for both auxiliary and primary task can be found respectively in `config/auxiliary.yaml` and `config/primary.yaml`. All the models generated in the target task training are saved in `models/primary`, while the intermediate pretrained models are placed inside `models/auxiliary` folder.

## Datasets 📁
- RWF-2000: [Box Drive](https://duke.app.box.com/s/kfgnl5bfy7w75cngopskms8kbh5w1mvu), unzip and run `python tools/rwf_2000_dataset_creation.py` <- This second
- HMDB51: run `python tools/hmdb51_download.py` <- This first
- UCF101: run `python tools/ucf101_download.py`
- UCF-Crime: [Dropbox](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0), unrar and run `python tools/ucf_crime_dataset_creation.py`

### Primary Task (without SSL pretraining) 🎯
1. `python primary.py --model_name model_primary`

### Auxiliary SSL Task + Primary Task 🧩
1. `python auxiliary.py --model_name model_ssl`
2. `python primary.py --model_name=model_primary --model_ssl_rgb_flow model_ssl`

## Evaluation 📊
1. `python primary.py --eval --model_name=model_primary`

## Inference
1. `python infer.py`