import json
import yaml
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def load_json(path: str):
    """从 JSON 文件加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path: str):
    """将对象保存为 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_yaml(path: str):
    """从 YAML 文件加载配置"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_logger(name: str, level: int = logging.INFO):
    """创建带 StreamHandler 的 Logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# 基础指标
def calc_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calc_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def calc_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float('nan')

# 多任务综合评估
def compute_multitask_metrics(outputs: dict, labels: dict) -> dict:
    """
    计算多任务指标，支持:
      - outputs['curvature'] (N,3), labels['curvature'] (N,)
      - outputs['alignment'] (N,2), labels['alignment'] (N,)
      - outputs['disc'] (N,5,3),   labels['disc']   (N,5)
      - outputs['canal'] (N,11,2),  labels['canal']  (N,11)
    返回 dict:
      curvature_acc, curvature_f1,
      alignment_acc, alignment_f1, alignment_auc,
      disc_f1, canal_f1, canal_auc
    """
    metrics = {}
    # 曲度
    cur_logits = outputs['curvature']
    cur_true = labels['curvature']
    cur_pred = np.argmax(cur_logits, axis=1)
    metrics['curvature_acc'] = calc_accuracy(cur_true, cur_pred)
    metrics['curvature_f1']  = calc_macro_f1(cur_true, cur_pred)

    # 对齐
    align_logits = outputs['alignment']
    align_true = labels['alignment']
    align_pred = np.argmax(align_logits, axis=1)
    metrics['alignment_acc'] = calc_accuracy(align_true, align_pred)
    metrics['alignment_f1']  = calc_macro_f1(align_true, align_pred)
    # AUC 用正类概率
    align_prob = align_logits[:,1]
    metrics['alignment_auc'] = calc_auc(align_true, align_prob)

    # 椎间盘 (三级分类) - flatten
    disc_logits = outputs['disc'].reshape(-1, outputs['disc'].shape[-1])  # (N*5,3)
    disc_true = labels['disc'].reshape(-1)
    disc_pred = np.argmax(disc_logits, axis=1)
    metrics['disc_f1'] = calc_macro_f1(disc_true, disc_pred)

    # 椎管 (二分类) - flatten
    canal_logits = outputs['canal'].reshape(-1, outputs['canal'].shape[-1])  # (N*11,2)
    canal_true = labels['canal'].reshape(-1)
    canal_pred = np.argmax(canal_logits, axis=1)
    canal_prob = canal_logits[:,1]
    metrics['canal_f1']  = calc_macro_f1(canal_true, canal_pred)
    metrics['canal_auc'] = calc_auc(canal_true, canal_prob)

    return metrics
