# LeWorldModel Docker 使用指南

## 快速开始

### 1. 构建 Docker 镜像

```bash
cd /storage/le-wm
docker build -t le-wm:latest .
```

### 2. 启动容器

```bash
./dev_start.sh
```

### 3. 进入容器

```bash
./dev_into.sh
```

## 环境测试

进入容器后，运行测试脚本验证环境：

```bash
python test_model.py
```

## 训练模型

### 数据准备

1. 从 [HuggingFace](https://huggingface.co/collections/quentinll/lewm) 下载数据集
2. 解压到 `.stable-wm/` 目录：
   ```bash
   tar --zstd -xvf archive.tar.zst -C .stable-wm/
   ```

### 配置 WandB

编辑 `config/train/lewm.yaml`：

```yaml
wandb:
  enabled: True
  config:
    entity: your_entity
    project: your_project
```

### 运行训练

```bash
# 训练 pusht 任务
python train.py data=pusht

# 训练其他任务
python train.py data=tworoom
python train.py data=cube
```

## 评估模型

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

## 容器管理

### 查看容器状态

```bash
docker ps
```

### 停止容器

```bash
docker stop le-wm-container
```

### 删除容器

```bash
docker rm le-wm-container
```

### 查看容器日志

```bash
docker logs le-wm-container
```

## 目录结构

- `/workspace/le-wm` - 项目代码
- `/workspace/.stable-wm` - 数据和模型存储
- `/storage` - 主机存储目录

## 端口映射

- `8888` - JupyterLab
- `6006` - TensorBoard

## 启动 Jupyter

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

然后在浏览器访问 `http://localhost:8888`

## 启动 TensorBoard

```bash
tensorboard --logdir=/workspace/.stable-wm --bind_all
```

然后在浏览器访问 `http://localhost:6006`
