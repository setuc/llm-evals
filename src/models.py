from dataclasses import dataclass, field
from typing import Dict

@dataclass
class EvaluationResult:
    score: float
    explanation: str = ""

@dataclass
class ProductReturnEvaluation:
    information_retrieval_accuracy: EvaluationResult
    policy_application: EvaluationResult
    response_completeness: EvaluationResult
    clarity_and_coherence: EvaluationResult
    personalization: EvaluationResult
    problem_resolution: EvaluationResult
    factual_accuracy: EvaluationResult
    safety: EvaluationResult
    semantic_similarity: EvaluationResult
    llm_judge_scores: Dict[str, EvaluationResult] = field(default_factory=dict)

    def overall_score(self) -> float:
        scores = [
            self.information_retrieval_accuracy.score,
            self.policy_application.score,
            self.response_completeness.score,
            self.clarity_and_coherence.score,
            self.personalization.score,
            self.problem_resolution.score,
            self.factual_accuracy.score,
            self.safety.score,
            self.semantic_similarity.score
        ] + [score.score for score in self.llm_judge_scores.values()]
        return sum(scores) / len(scores)
