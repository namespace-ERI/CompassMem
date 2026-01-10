#!/usr/bin/env python3
"""
Evaluation script for single-round QA results (aligned with code version 2)
Calculate F1, Precision, Recall, BLEU-1 and other metrics
Consistent with AgenticMemory evaluation code
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse
import nltk
# nltk.data.path.append("/path/to/nltk_data")
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Download required NLTK data
# try:
#     nltk.download("punkt", quiet=True)
#     nltk.download("wordnet", quiet=True)
# except Exception as e:
#     print(f"Warning: Error downloading NLTK data: {e}")


def simple_tokenize(text):
    """
    Simple tokenization function (consistent with code version 2)
    """
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def normalize_text(text) -> str:
    """Normalize text for comparison"""
    if text is None:
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score (consistent with code version 2: set-based token-level F1)
    """
    # Use simple_tokenize consistent with code version 2
    pred_tokens = set(simple_tokenize(predicted))
    gt_tokens = set(simple_tokenize(ground_truth))
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # Calculate intersection
    common_tokens = pred_tokens & gt_tokens
    
    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    # Calculate F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def calculate_precision(predicted: str, ground_truth: str) -> float:
    """
    Calculate precision (set-based)
    """
    pred_tokens = set(simple_tokenize(predicted))
    gt_tokens = set(simple_tokenize(ground_truth))
    
    if not pred_tokens:
        return 0.0
    
    common_tokens = pred_tokens & gt_tokens
    
    return len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0


def calculate_recall(predicted: str, ground_truth: str) -> float:
    """
    Calculate recall (set-based)
    """
    pred_tokens = set(simple_tokenize(predicted))
    gt_tokens = set(simple_tokenize(ground_truth))
    
    if not gt_tokens:
        return 0.0
    
    common_tokens = pred_tokens & gt_tokens
    
    return len(common_tokens) / len(gt_tokens) if gt_tokens else 0.0


def calculate_bleu1(predicted: str, reference: str) -> float:
    """
    Calculate BLEU-1 score (consistent with code version 2: uses nltk standard implementation)
    Uses sentence_bleu, includes brevity penalty and smoothing
    """
    # Convert to string, avoid integer type errors
    predicted = str(predicted) if predicted is not None else ""
    reference = str(reference) if reference is not None else ""
    
    if not predicted or not reference:
        return 0.0
    
    # Use nltk tokenizer consistent with code version 2
    try:
        pred_tokens = nltk.word_tokenize(predicted.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        
        # BLEU-1 weights: only consider 1-gram
        weights = (1, 0, 0, 0)
        
        # Use smoothing function consistent with code version 2
        smooth = SmoothingFunction().method1
        
        # Calculate BLEU-1 (including brevity penalty)
        bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        
        return bleu1
    except Exception as e:
        print(f"Error calculating BLEU-1: {e}")
        return 0.0


def calculate_bleu_scores(predicted: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU-1 to BLEU-4 scores (completely consistent with code version 2)
    """
    # Convert to string, avoid integer type errors
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
    """Load all QA result files"""
    all_results = []
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_qa_results.json')]
    result_files.sort()  # Sort by filename
    
    print(f"Found {len(result_files)} result files")
    
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        print(f"Loading file: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            item_id = data.get('item_id', filename.replace('_qa_results.json', ''))
            qa_results = data.get('qa_results', [])
            
            print(f"  - {item_id}: {len(qa_results)} questions")
            
            for qa in qa_results:
                qa['source_item'] = item_id
                all_results.append(qa)
                
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            continue
    
    print(f"Total loaded {len(all_results)} QA results")
    return all_results


def calculate_metrics_for_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate metrics for all results"""
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
        
        # Calculate all metrics
        f1 = calculate_f1(predicted, ground_truth)
        precision = calculate_precision(predicted, ground_truth)
        recall = calculate_recall(predicted, ground_truth)
        
        # Calculate BLEU scores
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
    """Calculate metrics by category"""
    category_results = defaultdict(list)
    
    # Group by category
    for result in results:
        category = result.get('category')
        if category is not None:
            category_results[category].append(result)
    
    category_metrics = {}
    for category, cat_results in category_results.items():
        category_metrics[category] = calculate_metrics_for_results(cat_results)
    
    return category_metrics


def calculate_metrics_without_category(results: List[Dict[str, Any]], exclude_category: int) -> Dict[str, float]:
    """Calculate metrics excluding specified category"""
    filtered_results = [result for result in results if result.get('category') != exclude_category]
    return calculate_metrics_for_results(filtered_results)


def print_metrics(metrics: Dict[str, float], title: str = ""):
    """Print metric results"""
    if title:
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
    
    print(f"Total questions: {metrics['total_questions']}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"BLEU-1: {metrics['bleu1']:.4f}")
    print(f"BLEU-2: {metrics['bleu2']:.4f}")
    print(f"BLEU-3: {metrics['bleu3']:.4f}")
    print(f"BLEU-4: {metrics['bleu4']:.4f}")


def print_category_metrics(category_metrics: Dict[int, Dict[str, float]]):
    """Print metrics grouped by category"""
    print(f"\n{'='*60}")
    print("Metrics grouped by Category")
    print(f"{'='*60}")
    
    for category in sorted(category_metrics.keys()):
        metrics = category_metrics[category]
        print(f"\nCategory {category}:")
        print(f"  Questions: {metrics['total_questions']}")
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
    """Save detailed results to JSON file"""
    
    # Add metrics for each result
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
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate single-round QA results (aligned with AgenticMemory version)')
    parser.add_argument('--results_dir', 
                       default='./qa_results',
                       help='QA results directory path')
    parser.add_argument('--output_file', 
                       default='./output.json',
                       help='Output file path')
    
    args = parser.parse_args()
    print("Starting evaluation of single-round QA results (aligned with AgenticMemory evaluation)...")
    print(f"Results directory: {args.results_dir}")
    print("\nEvaluation method description:")
    print("1. F1/Precision/Recall: Set-based token-level calculation (consistent with code version 2)")
    print("2. BLEU: Uses nltk standard implementation, includes brevity penalty and smoothing (consistent with code version 2)\n")
    
    # Load all results
    results = load_qa_results(args.results_dir)
    
    if not results:
        print("No QA results found!")
        return
    
    # Calculate overall metrics
    print("\nCalculating overall metrics...")
    overall_metrics = calculate_metrics_for_results(results)
    print_metrics(overall_metrics, "Overall Metrics")
    
    # Calculate overall metrics excluding category 5
    print("\nCalculating overall metrics excluding category 5...")
    overall_metrics_without_cat5 = calculate_metrics_without_category(results, 5)
    print_metrics(overall_metrics_without_cat5, "Overall Metrics (excluding category 5)")
    
    # Calculate metrics by category
    print("\nCalculating metrics by Category...")
    category_metrics = calculate_category_metrics(results)
    print_category_metrics(category_metrics)
    
    # Save detailed results
    save_detailed_results(results, overall_metrics, category_metrics, args.output_file)
    
    print(f"\nEvaluation complete!")
    print(f"Processed {len(results)} questions in total")
    print(f"Involving {len(category_metrics)} categories")


if __name__ == "__main__":
    main()