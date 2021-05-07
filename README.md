# 基于深度学习的文本生成图片

## 预训练模型
运行脚本下载预训练模型 `bash scripts/download_models.sh`. 

## 运行
运行 `scripts/run_model.py` 来根据Json数据生成图片

```bash
python scripts/run_model.py \
  --checkpoint sg2im-models/vg128.pt \
  --scene_graphs scene_graphs/figure_6_sheep.json \
  --output_dir outputs
```
