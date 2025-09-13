import numpy as np
from scipy import stats
import pandas as pd

# داده‌های جدول
data = {
    'A': {
        'AUC': [0.996, 0.98, 0.982, 0.998, 0.99],
        'Recall': [96.87, 95.94, 97.65, 94.42, 96.14],
        'F1': [96.41, 97.32, 94.69, 98.19, 95.06]
    },
    'B': {
        'AUC': [0.968, 0.965, 0.953, 0.956, 0.971],
        'Recall': [91.21, 93.76, 94.15, 92.39, 91.08],
        'F1': [92.25, 94.11, 92.78, 93.49, 91.97]
    },
    'C': {
        'AUC': [0.914, 0.877, 0.871, 0.896, 0.878],
        'Recall': [87.06, 86.16, 85.43, 88.85, 87.62],
        'F1': [85.67, 87.96, 86.18, 85.72, 87.91]
    },
    'D': {
        'AUC': [0.783, 0.793, 0.816, 0.777, 0.801],
        'Recall': [80.58, 79.44, 77.68, 80.61, 77.57],
        'F1': [80.31, 77.95, 78.04, 77.49, 81.76]
    }
}

def perform_ttest(model1_name, model1_data, model2_name, model2_data):
    """انجام t-test بین دو مدل برای هر معیار"""
    results = {}
    
    for metric in ['AUC', 'Recall', 'F1']:
        t_stat, p_value = stats.ttest_ind(model1_data[metric], model2_data[metric])
        results[metric] = {
            'mean_diff': np.mean(model1_data[metric]) - np.mean(model2_data[metric]),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return results

# محاسبه میانگین‌ها
print("=== DESCRIPTIVE STATISTICS ===")
for model in ['A', 'B', 'C', 'D']:
    print(f"\nModel {model}:")
    for metric in ['AUC', 'Recall', 'F1']:
        mean_val = np.mean(data[model][metric])
        std_val = np.std(data[model][metric], ddof=1)
        print(f"  {metric}: Mean = {mean_val:.3f}, Std = {std_val:.3f}")

print("\n" + "="*60)
print("=== T-TEST RESULTS (Model A vs Others) ===")
print("="*60)

# مقایسه مدل A با سایر مدل‌ها
for other_model in ['B', 'C', 'D']:
    print(f"\n🔍 Model A vs Model {other_model}:")
    print("-" * 40)
    
    results = perform_ttest('A', data['A'], other_model, data[other_model])
    
    for metric in ['AUC', 'Recall', 'F1']:
        r = results[metric]
        significance = "✅ Significant" if r['significant'] else "❌ Not Significant"
        
        print(f"{metric:7}: Mean Diff = {r['mean_diff']:+7.3f}, "
              f"t = {r['t_statistic']:6.3f}, "
              f"p = {r['p_value']:.6f} ({significance})")

# خلاصه کلی
print("\n" + "="*60)
print("=== SUMMARY FOR PAPER ===")
print("="*60)

summary_results = {}
for other_model in ['B', 'C', 'D']:
    results = perform_ttest('A', data['A'], other_model, data[other_model])
    summary_results[other_model] = results

print("\nStatistical significance of Model A superiority:")
for other_model in ['B', 'C', 'D']:
    sig_metrics = []
    for metric in ['AUC', 'Recall', 'F1']:
        if summary_results[other_model][metric]['significant']:
            sig_metrics.append(metric)
    
    if sig_metrics:
        print(f"• vs Model {other_model}: Significantly better in {', '.join(sig_metrics)} (p < 0.05)")
    else:
        print(f"• vs Model {other_model}: No significant differences found")

