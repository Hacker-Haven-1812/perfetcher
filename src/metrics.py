import os
import time

class MetricsCalculator:
    def __init__(self):
        # We start with empty dicts to store the values
        self.raw_counts = {
            "total_prefetches": 0,
            "useful_prefetches": 0,
            "timely_prefetches": 0,
            "total_cache_misses": 0,
            "covered_misses": 0,
            "total_accesses": 0,
            "cache_hits": 0,
            "total_evictions": 0,
            "evictions_caused_by_prefetch": 0,
            "pollution": 0,
            "inference_calls": 0
        }
        
        self.confusion_matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.classification = {}
        self.effectiveness = {}
        self.resources = {
            "total_prefetch_bytes": 0,
            "useful_bytes": 0,
            "bandwidth_overhead_bytes": 0,
            "bandwidth_overhead_mb": 0.0,
            "model_size_mb": 0.0,
            "inference_time_per_prediction_ms": 0.0,
            "inference_cpu_usage_percent": 0.0,
            "memory_usage_mb": 0.0
        }
        self.performance = {
            "speedup_percent": 0.0,
            "cold_time_sec": 0.0,
            "prefetched_time_sec": 0.0,
            "cold_time_median_sec": 0.0,
            "prefetched_time_median_sec": 0.0
        }

    def compute_all_metrics(self):
        self._compute_confusion_matrix()
        self._compute_classification_metrics()
        self._compute_effectiveness()
        # Resources and Performance are populated externally mostly
        return {
            "raw_counts": self.raw_counts,
            "confusion_matrix": self.confusion_matrix,
            "classification_metrics": self.classification,
            "prefetch_effectiveness": self.effectiveness,
            "resources": self.resources,
            "performance": self.performance
        }
        
    def populate_from_evaluation(self, prefetched_set, actual_set, all_possible_files_count=1000):
        # prefetched_set: Set of file paths predicted by model
        # actual_set: Set of file paths actually opened by the application during launch
        
        prefetched = set(prefetched_set)
        actual = set(actual_set)
        
        # Calculate True Positives (Useful prefetches)
        tp = len(prefetched.intersection(actual))
        
        # Calculate False Positives (Overpredictions)
        fp = len(prefetched.difference(actual))
        
        # Calculate False Negatives (Missed opportunities)
        fn = len(actual.difference(prefetched))
        
        # Calculate True Negatives
        tn = all_possible_files_count - (tp + fp + fn)
        
        self.confusion_matrix = {"tp": tp, "fp": fp, "fn": fn, "tn": max(0, tn)}
        
        self.raw_counts["total_prefetches"] = len(prefetched)
        self.raw_counts["useful_prefetches"] = tp
        self.raw_counts["timely_prefetches"] = tp # Simplification: Assuming all useful were timely
        self.raw_counts["total_accesses"] = len(actual)
        self.raw_counts["pollution"] = fp # Predict not used -> wastes cache

    def _compute_confusion_matrix(self):
        # Already computed in populate_from_evaluation
        pass

    def _compute_classification_metrics(self):
        TP = self.confusion_matrix["tp"]
        FP = self.confusion_matrix["fp"]
        FN = self.confusion_matrix["fn"]
        TN = self.confusion_matrix["tn"]
        
        P = TP + FN
        N = TN + FP
        
        acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / P if P > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = TN / N if N > 0 else 0
        balanced_acc = (recall + specificity) / 2
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        fpr = FP / N if N > 0 else 0
        fnr = FN / P if P > 0 else 0
        
        self.classification = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "balanced_accuracy": balanced_acc,
            "negative_predictive_value": npv,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        }

    def _compute_effectiveness(self):
        TP = self.confusion_matrix["tp"]
        FP = self.confusion_matrix["fp"]
        total_accesses = max(self.raw_counts["total_accesses"], 1)
        total_prefetches = max(self.raw_counts["total_prefetches"], 1)
        
        coverage = TP / total_accesses
        hit_rate = TP / total_prefetches
        miss_rate = 1.0 - hit_rate
        overprediction_rate = FP / total_prefetches
        pollution_rate = FP / total_prefetches # Proportion of cache taken by unused objects
        prefetch_accuracy = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        self.effectiveness = {
            "coverage": coverage,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "timeliness": min(1.0, coverage * 1.1), # Approximation
            "overprediction_rate": overprediction_rate,
            "pollution_rate": pollution_rate,
            "prefetch_accuracy": prefetch_accuracy
        }
