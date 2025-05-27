>📋  A template README.md for code accompanying a Machine Learning paper

# BeliefMapNav: 3D Voxel-Based Belief Map for Zero-Shot Object Navigation

## Requirements

To install requirements:

```setup
conda_env_name=beliefmap
conda create -n $conda_env_name python=3.9 -y
conda activate $conda_env_name
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2
pip install -r requirements.txt
pip install -e .[habitat]
pip install -e .[reality]
```
#### [Whether you're using conda or not]
Clone the following repo within this one (simply cloning will suffice):
```bash
git clone git@github.com:WongKinYiu/yolov7.git
```

## :dart: Downloading the HM3D dataset

### Matterport
First, set the following variables during installation (don't need to put in .bashrc):
```bash
MATTERPORT_TOKEN_ID=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
MATTERPORT_TOKEN_SECRET=<FILL IN FROM YOUR ACCOUNT INFO IN MATTERPORT>
DATA_DIR=</path/to/BliefmapNav/data>

HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
```

### Clone and install habitat-lab, then download datasets
*Ensure that the correct conda environment is activated!!*
```bash
# Download HM3D 3D scans (scenes_dataset)
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR &&

# Download HM3D ObjectNav dataset episodes
wget $HM3D_OBJECTNAV &&
unzip objectnav_hm3d_v1.zip &&
mkdir -p $DATA_DIR/datasets/objectnav/hm3d  &&
mv objectnav_hm3d_v1 $DATA_DIR/datasets/objectnav/hm3d/v1 &&
rm objectnav_hm3d_v1.zip
```

## :weight_lifting: Downloading weights for various models
The weights for MobileSAM, GroundingDINO, and PointNav must be saved to the `data/` directory. The weights can be downloaded from the following links:
- `mobile_sam.pt`:  https://github.com/ChaoningZhang/MobileSAM
- `groundingdino_swint_ogc.pth`: https://github.com/IDEA-Research/GroundingDINO
- `yolov7-e6e.pt`: https://github.com/WongKinYiu/yolov7
- `pointnav_weights.pth`: included inside the [data](data) subdirectory

## Evaluation
```eval
conda activate beliefmap
./scripts/launch_multi_eval.sh
```

## Results

Our model achieves the following performance on :

| Method                            | Unsupervised | Zero-shot | HM3D SR↑ | HM3D SPL↑ | MP3D SR↑ | MP3D SPL↑ | HSSD SR↑ | HSSD SPL↑ |
|----------------------------------|--------------|-----------|----------|-----------|----------|-----------|----------|-----------|
| Habitat-Web [ramrakhya2022habitat]      | ❌           | ❌        | 41.5     | 16.0      | 31.6     | 8.5       | -        | -         |
| OVRL [yadav2023offline]                    | ❌           | ❌        | -        | -         | 28.6     | 7.4       | -        | -         |
| ProcTHOR [deitke2022️]                     | ❌           | ❌        | 54.4     | 31.8      | -        | -         | -        | -         |
| SGM [zhang2024imagine]                     | ❌           | ❌        | 60.2     | 30.8      | 37.7     | 14.7      | -        | -         |
| **---**                              |              |           |          |           |          |           |          |           |
| ZSON [majumdar2022zson]                    | ❌           | ✔         | 25.5     | 12.6      | 15.3     | 4.8       | -        | -         |
| PSL [sun2024prioritized]                    | ❌           | ✔         | 42.4     | 19.2      | 18.9     | 6.4       | -        | -         |
| PixNav [cai2024bridging]                    | ❌           | ✔         | 37.9     | 20.5      | -        | -         | -        | -         |
| **---**                              |              |           |          |           |          |           |          |           |
| VLFM [yokoyama2024vlfm]                     | ✔            | ✔         | 52.5     | 30.4      | 36.4     | 17.5      | -        | -         |
| ESC [zhou2023esc]                       | ✔            | ✔         | 39.2     | 22.3      | 28.7     | 14.2      | 38.1     | 22.2      |
| Cows [gadre2023cows]                       | ✔            | ✔         | -        | -         | 9.2      | 4.9       |          |           |
| L3MVN [yu2023l3mvn]{}                     | ✔            | ✔         | 50.4     | 23.1      | 34.9     | 14.5      | 41.2     | 22.5      |
| ImagineNav [zhao2024imaginenav]{}          | ✔            | ✔         | 53.0     | 23.8      | -        | -         | 51.0     | 24.9      |
| VoroNav [wu2024voronav]                     | ✔            | ✔         | 42.0     | 26.0      | -        | -         | 41.0     | 23.2      |
| GAMap [huang2024gamap]                     | ✔            | ✔         | 53.1     | 26.0      | -        | -         | -        | -         |
| OpenFMNav [kuang2024openfmnav]              | ✔            | ✔         | 52.5     | 24.1      | 37.2     | 15.7      | -        | -         |
| InstructNav [long2024instructnav]            | ✔            | ✔         | 58.0     | 20.9      | -        | -         | -        | -         |
| **BeliefMapNav**                    | ✔            | ✔         | **61.4** | **30.6**  | **37.3** | **17.6**  | **65.2** | **32.1**  |

## :newspaper: License

BliefmapNav is released under the [MIT License](LICENSE).
