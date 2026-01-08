#!/usr/bin/env python3
"""
评估单轮QA结果的指标计算脚本（对齐代码二版本）
计算F1、Precision、Recall、BLEU-1等指标
与AgenticMemory评测代码保持一致
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse
import nltk
nltk.data.path.append("/share/project/zyt/envs/nltk/nltk_data")
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Download required NLTK data
# try:
#     nltk.download("punkt", quiet=True)
#     nltk.download("wordnet", quiet=True)
# except Exception as e:
#     print(f"Warning: Error downloading NLTK data: {e}")


def simple_tokenize(text):
    """
    Simple tokenization function (与代码二保持一致)
    """
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def normalize_text(text) -> str:
    """标准化文本，用于比较"""
    if text is None:
        return ""
    
    # 转换为字符串
    text = str(text)
    
    # 转换为小写
    text = text.lower().strip()
    
    # 移除标点符号和多余空格
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """
    计算F1分数（与代码二保持一致：基于集合的token-level F1）
    """
    # 使用simple_tokenize与代码二保持一致
    pred_tokens = set(simple_tokenize(predicted))
    gt_tokens = set(simple_tokenize(ground_truth))
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # 计算交集
    common_tokens = pred_tokens & gt_tokens
    
    # 计算precision和recall
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    # 计算F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def calculate_precision(predicted: str, ground_truth: str) -> float:
    """
    计算精确率（基于集合）
    """
    pred_tokens = set(simple_tokenize(predicted))
    gt_tokens = set(simple_tokenize(ground_truth))
    
    if not pred_tokens:
        return 0.0
    
    common_tokens = pred_tokens & gt_tokens
    
    return len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0


def calculate_recall(predicted: str, ground_truth: str) -> float:
    """
    计算召回率（基于集合）
    """
    pred_tokens = set(simple_tokenize(predicted))
    gt_tokens = set(simple_tokenize(ground_truth))
    
    if not gt_tokens:
        return 0.0
    
    common_tokens = pred_tokens & gt_tokens
    
    return len(common_tokens) / len(gt_tokens) if gt_tokens else 0.0


def calculate_bleu1(predicted: str, reference: str) -> float:
    """
    计算BLEU-1分数（与代码二保持一致：使用nltk标准实现）
    使用sentence_bleu，包含brevity penalty和smoothing
    """
    # 转换为字符串，避免整数类型导致的错误
    predicted = str(predicted) if predicted is not None else ""
    reference = str(reference) if reference is not None else ""
    
    if not predicted or not reference:
        return 0.0
    
    # 使用nltk tokenizer与代码二保持一致
    try:
        pred_tokens = nltk.word_tokenize(predicted.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        
        # BLEU-1权重：只考虑1-gram
        weights = (1, 0, 0, 0)
        
        # 使用smoothing function与代码二保持一致
        smooth = SmoothingFunction().method1
        
        # 计算BLEU-1（包含brevity penalty）
        bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        
        return bleu1
    except Exception as e:
        print(f"Error calculating BLEU-1: {e}")
        return 0.0


def calculate_bleu_scores(predicted: str, reference: str) -> Dict[str, float]:
    """
    计算BLEU-1到BLEU-4分数（与代码二完全一致）
    """
    # 转换为字符串，避免整数类型导致的错误
    predicted = str(predicted) if predicted is not None else ""
    reference = str(reference) if reference is not None else ""
    
    if not predicted or not reference:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    try:
        pred_tokens = nltk.word_tokenize(predicted.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        
        weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
        smooth = SmoothingFunction().method1
        
        scores = {}
        for n, weights in enumerate(weights_list, start=1):
            try:
                score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
            except Exception as e:
                print(f"Error calculating BLEU-{n} score: {e}")
                score = 0.0
            scores[f"bleu{n}"] = score
        
        return scores
    except Exception as e:
        print(f"Error in calculate_bleu_scores: {e}")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}


def load_qa_results(results_dir: str) -> List[Dict[str, Any]]:
    """加载所有QA结果文件"""
    all_results = []
    
    # 获取所有结果文件
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_qa_results.json')]
    result_files.sort()  # 按文件名排序
    
    print(f"找到 {len(result_files)} 个结果文件")
    
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        print(f"加载文件: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            item_id = data.get('item_id', filename.replace('_qa_results.json', ''))
            qa_results = data.get('qa_results', [])
            
            print(f"  - {item_id}: {len(qa_results)} 个问题")
            
            for qa in qa_results:
                qa['source_item'] = item_id
                all_results.append(qa)
                
        except Exception as e:
            print(f"错误加载文件 {filename}: {e}")
            continue
    
    print(f"总共加载了 {len(all_results)} 个QA结果")
    return all_results


def calculate_metrics_for_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算所有结果的指标"""
    if not results:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'bleu1': 0.0,
            'bleu2': 0.0,
            'bleu3': 0.0,
            'bleu4': 0.0,
            'total_questions': 0
        }
    
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_bleu1 = 0
    total_bleu2 = 0
    total_bleu3 = 0
    total_bleu4 = 0
    
    for result in results:
        predicted = result.get('answer', '')
        ground_truth = result.get('ground_truth', '')
        
        # 计算各项指标
        f1 = calculate_f1(predicted, ground_truth)
        precision = calculate_precision(predicted, ground_truth)
        recall = calculate_recall(predicted, ground_truth)
        
        # 计算BLEU分数
        bleu_scores = calculate_bleu_scores(predicted, ground_truth)
        
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        total_bleu1 += bleu_scores['bleu1']
        total_bleu2 += bleu_scores['bleu2']
        total_bleu3 += bleu_scores['bleu3']
        total_bleu4 += bleu_scores['bleu4']
    
    num_questions = len(results)
    
    return {
        'f1': total_f1 / num_questions,
        'precision': total_precision / num_questions,
        'recall': total_recall / num_questions,
        'bleu1': total_bleu1 / num_questions,
        'bleu2': total_bleu2 / num_questions,
        'bleu3': total_bleu3 / num_questions,
        'bleu4': total_bleu4 / num_questions,
        'total_questions': num_questions
    }


def calculate_category_metrics(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """按category计算指标"""
    category_results = defaultdict(list)
    
    # 按category分组
    for result in results:
        category = result.get('category')
        if category is not None:
            category_results[category].append(result)
    
    category_metrics = {}
    for category, cat_results in category_results.items():
        category_metrics[category] = calculate_metrics_for_results(cat_results)
    
    return category_metrics


def calculate_metrics_without_category(results: List[Dict[str, Any]], exclude_category: int) -> Dict[str, float]:
    """计算排除指定category的指标"""
    filtered_results = [result for result in results if result.get('category') != exclude_category]
    return calculate_metrics_for_results(filtered_results)


def print_metrics(metrics: Dict[str, float], title: str = ""):
    """打印指标结果"""
    if title:
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
    
    print(f"总问题数: {metrics['total_questions']}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"BLEU-1: {metrics['bleu1']:.4f}")
    print(f"BLEU-2: {metrics['bleu2']:.4f}")
    print(f"BLEU-3: {metrics['bleu3']:.4f}")
    print(f"BLEU-4: {metrics['bleu4']:.4f}")


def print_category_metrics(category_metrics: Dict[int, Dict[str, float]]):
    """打印按category分组的指标"""
    print(f"\n{'='*60}")
    print("按Category分组的指标")
    print(f"{'='*60}")
    
    for category in sorted(category_metrics.keys()):
        metrics = category_metrics[category]
        print(f"\nCategory {category}:")
        print(f"  问题数: {metrics['total_questions']}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  BLEU-1: {metrics['bleu1']:.4f}")
        print(f"  BLEU-2: {metrics['bleu2']:.4f}")
        print(f"  BLEU-3: {metrics['bleu3']:.4f}")
        print(f"  BLEU-4: {metrics['bleu4']:.4f}")


def save_detailed_results(results: List[Dict[str, Any]], 
                         overall_metrics: Dict[str, float],
                         category_metrics: Dict[int, Dict[str, float]],
                         output_file: str):
    """保存详细结果到JSON文件"""
    
    # 为每个结果添加指标
    detailed_results = []
    for result in results:
        predicted = result.get('answer', '')
        ground_truth = result.get('ground_truth', '')
        
        detailed_result = result.copy()
        bleu_scores = calculate_bleu_scores(predicted, ground_truth)
        
        detailed_result['metrics'] = {
            'f1': calculate_f1(predicted, ground_truth),
            'precision': calculate_precision(predicted, ground_truth),
            'recall': calculate_recall(predicted, ground_truth),
            'bleu1': bleu_scores['bleu1'],
            'bleu2': bleu_scores['bleu2'],
            'bleu3': bleu_scores['bleu3'],
            'bleu4': bleu_scores['bleu4']
        }
        detailed_results.append(detailed_result)
    
    output_data = {
        'overall_metrics': overall_metrics,
        'overall_metrics_without_category5': calculate_metrics_without_category(results, 5),
        'category_metrics': category_metrics,
        'detailed_results': detailed_results,
        'summary': {
            'total_questions': len(results),
            'total_categories': len(category_metrics),
            'categories': list(category_metrics.keys())
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估单轮QA结果（对齐AgenticMemory版本）')
    parser.add_argument('--results_dir', 
                       default='/share/project/zyt/hyy/Memory/QA_system/output/qa_results_hierarchical_v3_two_stage_20260104_191206',
                       help='QA结果目录路径')
    parser.add_argument('--output_file', 
                       default='/share/project/zyt/hyy/Memory/QA_system/test_output/qa_results_hierarchical_v3_two_stage_20260104_191514.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    print("开始评估单轮QA结果（对齐AgenticMemory评测）...")
    print(f"结果目录: {args.results_dir}")
    print("\n评测方法说明:")
    print("1. F1/Precision/Recall: 基于集合的token-level计算（与代码二一致）")
    print("2. BLEU: 使用nltk标准实现，包含brevity penalty和smoothing（与代码二一致）\n")
    
    # 加载所有结果
    results = load_qa_results(args.results_dir)
    
    if not results:
        print("没有找到任何QA结果!")
        return
    
    # 计算总体指标
    print("\n计算总体指标...")
    overall_metrics = calculate_metrics_for_results(results)
    print_metrics(overall_metrics, "总体指标")
    
    # 计算排除第5类的总体指标
    print("\n计算排除第5类的总体指标...")
    overall_metrics_without_cat5 = calculate_metrics_without_category(results, 5)
    print_metrics(overall_metrics_without_cat5, "总体指标(排除第5类)")
    
    # 计算按category的指标
    print("\n计算按Category的指标...")
    category_metrics = calculate_category_metrics(results)
    print_category_metrics(category_metrics)
    
    # 保存详细结果
    save_detailed_results(results, overall_metrics, category_metrics, args.output_file)
    
    print(f"\n评估完成!")
    print(f"总共处理了 {len(results)} 个问题")
    print(f"涉及 {len(category_metrics)} 个categories")


if __name__ == "__main__":
    main()