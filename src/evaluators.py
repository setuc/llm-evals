import re
import random
from typing import Dict, Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from models import EvaluationResult, ProductReturnEvaluation

class LLMJudge:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.name = model_name

    def evaluate(self, query: str, context: str, response: str) -> EvaluationResult:
        prompt = f"""
            As an AI assistant, evaluate the following response.

            Query: {query}
            Context: {context}

            Criteria:
            1. Accuracy of information
            2. Completeness of the answer
            3. Relevance to the query
            4. Clarity and coherence

            Provide your evaluation in the following format:
            Score: <a number between 0 and 1>
            Explanation: <brief explanation>
            
            Response: {response}
            """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        score_match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", result)

        score = float(score_match.group(1)) if score_match else 0.5
        # Add some randomness to the score
        score = max(0, min(1, score + random.uniform(-0.2, 0.2)))
        
        explanation_match = re.search(r"Explanation:\s*(.*)", result, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = "No explanation provided."

        
        return EvaluationResult(score, explanation)

class RandomJudge:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, query: str, context: str, response: str) -> EvaluationResult:
        score = random.uniform(0, 1)
        explanation = f"Random score generated: {score:.2f}"
        return EvaluationResult(score, explanation)

def semantic_similarity(response: str, context: str) -> EvaluationResult:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    response_embedding = model.encode(response, convert_to_tensor=True)
    context_embedding = model.encode(context, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(response_embedding, context_embedding).item()
    return EvaluationResult(similarity, "Semantic similarity between response and context")

def evaluate_product_return_handling(
    response: str, 
    query: str, 
    context: str, 
    fact_checker: Callable, 
    toxicity_checker: Callable,
    llm_judges: Dict[str, LLMJudge]
) -> ProductReturnEvaluation:
    def score_with_explanation(score: float, explanation: str) -> EvaluationResult:
        return EvaluationResult(score, explanation)

    # Information Retrieval Accuracy
    product_mentioned = any(product in response.lower() for product in ["laptop", "smartphone", "book"])
    policy_mentioned = "return policy" in response.lower()
    purchase_history = re.search(r"\d{4}-\d{2}-\d{2}", response) is not None
    info_retrieval_score = (product_mentioned + policy_mentioned + purchase_history) / 3
    info_retrieval_explanation = f"Product mentioned: {product_mentioned}, Policy mentioned: {policy_mentioned}, Purchase history used: {purchase_history}"

    # Policy Application
    return_window_mentioned = any(term in response.lower() for term in ["30 days", "14 days", "7 days"])
    refund_calculation = "refund" in response.lower()
    exceptions_mentioned = "defective" in response.lower() if "defective" in query.lower() else True
    policy_application_score = (return_window_mentioned + refund_calculation + exceptions_mentioned) / 3
    policy_application_explanation = f"Return window mentioned: {return_window_mentioned}, Refund calculation: {refund_calculation}, Exceptions handled: {exceptions_mentioned}"

    # Response Completeness
    steps_provided = "steps" in response.lower() or re.search(r"\d+\)", response) is not None
    shipping_info = "shipping" in response.lower() or "drop off" in response.lower()
    timeline_explained = "5-7 business days" in response or "3-5 business days" in response
    completeness_score = (steps_provided + shipping_info + timeline_explained) / 3
    completeness_explanation = f"Steps provided: {steps_provided}, Shipping info: {shipping_info}, Timeline explained: {timeline_explained}"

    # Clarity and Coherence
    easy_to_understand = len(response.split()) < 150  # Assuming shorter responses are clearer
    appropriate_tone = "thank you" in response.lower() or "please" in response.lower()
    logical_order = steps_provided  # Reusing this as a proxy for logical order
    clarity_score = (easy_to_understand + appropriate_tone + logical_order) / 3
    clarity_explanation = f"Easy to understand: {easy_to_understand}, Appropriate tone: {appropriate_tone}, Logical order: {logical_order}"

    # Personalization
    customer_name_used = any(name in response for name in ["John", "Doe"])
    past_interaction = purchase_history  # Reusing this as a proxy for referencing past interactions
    specific_situation = product_mentioned and return_window_mentioned
    personalization_score = (customer_name_used + past_interaction + specific_situation) / 3
    personalization_explanation = f"Customer name used: {customer_name_used}, Past interaction referenced: {past_interaction}, Specific situation addressed: {specific_situation}"

    # Problem Resolution
    fully_addressed = all([product_mentioned, policy_mentioned, steps_provided, refund_calculation])
    follow_up_addressed = "if you have any questions" in response.lower()
    alternatives_provided = "alternative" in response.lower() or "option" in response.lower()
    resolution_score = (fully_addressed + follow_up_addressed + alternatives_provided) / 3
    resolution_explanation = f"Fully addressed: {fully_addressed}, Follow-up addressed: {follow_up_addressed}, Alternatives provided: {alternatives_provided}"

    # Factual Accuracy
    factual_accuracy_score = fact_checker(response)
    factual_accuracy_explanation = "Factual accuracy checked using external fact-checking model"

    # Safety
    safety_score = 1 - toxicity_checker(response)  # Invert the toxicity score
    safety_explanation = "Safety score based on toxicity check"

    # Semantic Similarity
    similarity_score = semantic_similarity(response, context)

    # LLM judges evaluation
    llm_judge_scores = {
        name: judge.evaluate(query, context, response)
        for name, judge in llm_judges.items()
    }

    return ProductReturnEvaluation(
        information_retrieval_accuracy=score_with_explanation(info_retrieval_score, info_retrieval_explanation),
        policy_application=score_with_explanation(policy_application_score, policy_application_explanation),
        response_completeness=score_with_explanation(completeness_score, completeness_explanation),
        clarity_and_coherence=score_with_explanation(clarity_score, clarity_explanation),
        personalization=score_with_explanation(personalization_score, personalization_explanation),
        problem_resolution=score_with_explanation(resolution_score, resolution_explanation),
        factual_accuracy=score_with_explanation(factual_accuracy_score, factual_accuracy_explanation),
        safety=score_with_explanation(safety_score, safety_explanation),
        semantic_similarity=similarity_score,
        llm_judge_scores=llm_judge_scores
    )