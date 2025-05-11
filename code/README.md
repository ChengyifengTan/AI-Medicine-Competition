# CervAI Project

## 目录结构

```
cervai_project/
├── data/
│   ├── train.json             # 训练集标签
│   ├── valid.json             # 验证集标签（新增）
│   ├── test.json              # 测试集列表
│   ├── test_predict.json      # 预测结果输出
│   └── images/                # 影像数据根目录
│       └── {id}/              # 每个样本文件夹
│           ├── sag/           # 矢状位切片（5.png,6.png,7.png）
│           └── tra/           # 横轴切片（2.png,2-3.png,...,7.png）
├── configs/
│   └── config.yaml            # 全局配置，包括 data、model、training、evaluation、logging
├── src/
│   ├── datasets.py            # 数据读取与预处理
│   ├── models.py              # 多切片多任务模型定义
│   ├── trainer.py             # 训练流程与验证指标计算
│   ├── utils.py               # 工具函数（加载配置、日志、指标）
│   └── predict.py             # 推理流程，生成 test_predict.json
├── train.py                   # 训练入口脚本
├── predict.py                 # 推理入口脚本
├── requirements.txt           # 环境依赖
└── README.md                  # 项目说明
```

## 环境依赖

```bash
pip install -r requirements.txt
```

## 数据准备

1. 将 `train.json`、`valid.json`、`test.json` 放入 `data/` 目录。
2. 将影像数据按样本 ID 放入 `data/images/{id}/sag/` 和 `data/images/{id}/tra/`。

   * `sag/` 中需包含：`5.png`,`6.png`,`7.png`
   * `tra/` 中需包含：`2.png`,`2-3.png`,`3.png`,… ,`6-7.png`,`7.png`

## 使用步骤

1. **训练模型**

   ```bash
   python train.py --config configs/config.yaml
   ```

   * 会在 `logging.save_dir` 中保存 `best_model.pt`

2. **生成预测**

   ```bash
   python predict.py --config configs/config.yaml
   ```

   * 推理结果保存在 `data/test_predict.json`

3. **评估结果**

   * 若需要在验证集上评估，可在 `trainer.py` 中自行调用 `evaluation` 部分的指标计算。

done
