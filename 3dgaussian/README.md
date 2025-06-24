# 3D Gaussian Splatting

算法代码与官方仓库保持一致
增加文件render_interp.py 用于基于训练好的模型插值生成新轨迹并渲染帧

### 用法
```bash
python render_interp.py \
    --model_path path/to/your/model \
    --iteration 30000 #指明使用的iter版本，忽略则默认最新\
    --view_indices 0 5 10 15 \
    --n_frames 60 \
    --output output/interp_path
