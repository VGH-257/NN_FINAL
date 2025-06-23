# NERF

## 训练前的数据准备

### 环境准备
```bash
conda create -n nerf python=3.10
conda activate nerf
```
- 安装[colmap](https://github.com/colmap/colmap)
- 安装[LLFF](https://github.com/Fyusion/LLFF)
- 配置[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

### 数据集
将视频转换后的图片放到 `./data/chair/images/`路径下

### 利用COLMAP估计相机参数
使用COLMAP得到相机参数：  
```bash
DATASET_PATH=./data/chair/

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse
```

转换成NeRF可读取的格式:  
根据NeRF作者的建议，使用LLFF提供的的imgs2poses.py文件获取所需相机参数。
```bash
python LLFF/imgs2poses.py ./data/chair/
```


### 训练
- 准备配置文件`./configs/chair.txt`。  
- 执行训练脚本：
```bash
python run_nerf.py --config configs/chair.txt
```

### 测试
```bash
python run_nerf.py --config configs/chair.txt --render_only --render_test
```