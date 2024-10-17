from typing import List, Dict, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)
from transformers import pipeline
from evaluators import LLMJudge, evaluate_product_return_handling, RandomJudge
from utils import generate_llm_response, calculate_iaa_metrics, calculate_judge_performance, print_evaluation_results

def run_evaluation(queries: List[str], contexts: List[str], gold_standard: List[float]) -> Tuple[Dict, List, Dict[str, float], Dict[str, Dict[str, float]]]:
    # Initialize models
    fact_checker = pipeline("text-classification", model="microsoft/deberta-base-mnli")
    toxicity_checker = pipeline("text-classification", model="unitary/toxic-bert")
    
    # Initialize LLM judges
    llm_judges = {
        "gpt2": LLMJudge("gpt2"),
        "distilgpt2": LLMJudge("distilgpt2"),
        "random_judge": RandomJudge("random_judge"),
        # Add more LLM judges as needed
    }

    # Generate responses
    responses = [generate_llm_response(query, context) for query, context in zip(queries, contexts)]

    # Prepare dataset for RAGAS
    dataset = Dataset.from_dict({
        "question": queries,
        "answer": responses,
        "contexts": [[context] for context in contexts],  # Each context needs to be a list
        "ground_truth": responses  # Using responses as ground truth for demonstration
    })

    # Define RAGAS evaluation metrics
    metrics = [faithfulness, answer_relevancy, context_recall]

    # Run RAGAS evaluation
    ragas_result = evaluate(dataset, metrics)

    # Run custom evaluation
    custom_evaluations = [
        evaluate_product_return_handling(
            response, query, context,
            lambda x: fact_checker(x)[0]['score'],
            lambda x: toxicity_checker(x)[0]['score'],
            llm_judges
        )
        for response, query, context in zip(responses, queries, contexts)
    ]

    # Calculate IAA metrics
    iaa_metrics = calculate_iaa_metrics(custom_evaluations)

    # Calculate judge performance
    judge_performance = calculate_judge_performance(custom_evaluations, gold_standard)

    return ragas_result, custom_evaluations, iaa_metrics, judge_performance

if __name__ == "__main__":
    # Sample data
    queries = [
        "I want to return the laptop I bought last month. What's the process?",
        "How do I return a defective smartphone?",
        "Can I get a refund for a book I ordered yesterday?"
    ]

    contexts = [
        "Customer John Doe purchased a laptop on 2023-05-15. Our return policy allows returns within 30 days of purchase for laptops. Refunds are processed within 5-7 business days after receiving the returned item.",
        "Smartphone return policy: Defective devices can be returned within 14 days of purchase. Customer must provide proof of purchase and describe the defect.",
        "Books can be returned within 7 days of delivery if unopened. Refunds are processed to the original payment method within 3-5 business days."
    ]

    # Define a mock gold standard for demonstration purposes
    gold_standard = [0.8, 0.7, 0.9]  # One score per query-response pair

    ragas_result, custom_evaluations, iaa_metrics, judge_performance = run_evaluation(queries, contexts, gold_standard)
    print_evaluation_results(ragas_result, custom_evaluations, iaa_metrics, judge_performance)