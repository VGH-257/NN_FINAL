# TensoRF

## 环境准备

- 安装colmap
- 安装[TensoRF](https://github.com/apchenstu/TensoRF)

## 数据准备

将视频转换成图片后放在`/data/chair`目录下

## 利用colmap估计相机参数

```bash
colmap auto_reconstructor \
	--image_path /path/to/your/images \
	--workspace_path /path/to/workspace

colmap model_converter \
	--input_path /path/to/your/workspace \
	--output_path /path/to/output \
	--output_type TXT
```

## 将colmap结果转化为nerf可接受的输入格式

利用TensoRF官方库提供的脚本`colmap2nerf.py` 进行转化

```bash
python colmap2nerf.py \
	--text /path/to/your/workspace \
	--images /path/to/your/images \
	--out /path/to/output
```

## 训练

- 准备配置文件`./configs/chair.txt`

- 运行官方库提供的脚本`train.py`

  ```bash
  python train.py --config configs/chair.txt
  ```

## 测试

```bash
python train.py \
	--config configs/chair.txt \
	--render_only 1 \
	--render_test 1
```

