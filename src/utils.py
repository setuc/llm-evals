from scipy import stats
import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff
from typing import List, Dict
from models import ProductReturnEvaluation

def calculate_iaa_metrics(evaluations: List[ProductReturnEvaluation]) -> Dict[str, float]:
    judge_scores = {judge: [eval.llm_judge_scores[judge].score for eval in evaluations] for judge in evaluations[0].llm_judge_scores}
    print(f'judge_scores: {judge_scores}')
    # Discretize scores into categories
    def discretize(score):
        if score < 0.33:
            return 0
        elif score < 0.67:
            return 1
        else:
            return 2

    discretized_scores = {judge: [discretize(score) for score in scores] for judge, scores in judge_scores.items()}
    
    # Cohen's Kappa (pairwise comparison)
    cohen_kappas = []
    judges = list(discretized_scores.keys())
    for i in range(len(judges)):
        for j in range(i+1, len(judges)):
            try:
                kappa = cohen_kappa_score(discretized_scores[judges[i]], discretized_scores[judges[j]])
                if not np.isnan(kappa):
                    cohen_kappas.append(kappa)
            except ValueError:
                pass
    
    # Krippendorff's Alpha
    try:
        reliability_data = np.array(list(judge_scores.values())).T
        alpha = krippendorff.alpha(reliability_data=reliability_data.tolist(), level_of_measurement='interval')
    except ValueError:
        alpha = None
    
    return {
        "avg_cohen_kappa": np.mean(cohen_kappas) if cohen_kappas else None,
        "krippendorff_alpha": alpha
    }

def calculate_judge_performance(evaluations: List[ProductReturnEvaluation], gold_standard: List[float]) -> Dict[str, Dict[str, float]]:
    judge_scores = {judge: [eval.llm_judge_scores[judge].score for eval in evaluations] for judge in evaluations[0].llm_judge_scores}
    
    performance = {}
    for judge, scores in judge_scores.items():
        precision = np.mean([1 if abs(s - g) < 0.1 else 0 for s, g in zip(scores, gold_standard)])
        recall = np.mean([1 if abs(s - g) < 0.1 else 0 for s, g in zip(scores, gold_standard)])
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        agreement = np.mean([1 if abs(s - g) < 0.1 else 0 for s, g in zip(scores, gold_standard)])
        z_score = stats.zscore(scores)
        
        performance[judge] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "percent_agreement": agreement,
            "avg_z_score": np.mean(z_score)
        }
    
    return performance

def generate_llm_response(query: str, context: str) -> str:
    # This is a mock function. In a real scenario, this would call your actual LLM.
    return "Thank you for contacting us about returning your product. Based on our records, you purchased a laptop on 2023-05-15. Our return policy allows returns within 30 days of purchase. To initiate the return process, please follow these steps: 1) Pack the laptop in its original packaging. 2) Include all accessories. 3) Print the return label from your account. 4) Drop off the package at any authorized shipping location. Once we receive the item, we'll process your refund within 5-7 business days. If you have any questions, please don't hesitate to ask."

def print_evaluation_results(ragas_result: Dict, custom_evaluations: List[ProductReturnEvaluation], iaa_metrics: Dict[str, float], judge_performance: Dict[str, Dict[str, float]]):
    print("RAGAS Evaluation Results:")
    print(ragas_result)

    print("\nCustom Evaluation Results:")
    for i, eval in enumerate(custom_evaluations):
        print(f"\nEvaluation for Response {i+1}:")
        print(f"Information Retrieval Accuracy: {eval.information_retrieval_accuracy.score:.2f}")
        print(f"  Explanation: {eval.information_retrieval_accuracy.explanation}")
        print(f"Policy Application: {eval.policy_application.score:.2f}")
        print(f"  Explanation: {eval.policy_application.explanation}")
        print(f"Response Completeness: {eval.response_completeness.score:.2f}")
        print(f"  Explanation: {eval.response_completeness.explanation}")
        print(f"Clarity and Coherence: {eval.clarity_and_coherence.score:.2f}")
        print(f"  Explanation: {eval.clarity_and_coherence.explanation}")
        print(f"Personalization: {eval.personalization.score:.2f}")
        print(f"  Explanation: {eval.personalization.explanation}")
        print(f"Problem Resolution: {eval.problem_resolution.score:.2f}")
        print(f"  Explanation: {eval.problem_resolution.explanation}")
        print(f"Factual Accuracy: {eval.factual_accuracy.score:.2f}")
        print(f"  Explanation: {eval.factual_accuracy.explanation}")
        print(f"Safety: {eval.safety.score:.2f}")
        print(f"  Explanation: {eval.safety.explanation}")
        print(f"Semantic Similarity: {eval.semantic_similarity.score:.2f}")
        print(f"  Explanation: {eval.semantic_similarity.explanation}")
        print("\nLLM Judge Scores:")
        for judge_name, judge_score in eval.llm_judge_scores.items():
            print(f"  {judge_name}: {judge_score.score:.2f}")
            print(f"    Explanation: {judge_score.explanation}")
        print(f"Overall Score: {eval.overall_score():.2f}")

    print("\nInter-Annotator Agreement Metrics:")
    for metric, value in iaa_metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Unable to calculate")

    print("\nJudge Performance Metrics:")
    for judge, metrics in judge_performance.items():
        print(f"\n{judge}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            