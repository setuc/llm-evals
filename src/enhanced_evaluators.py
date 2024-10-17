# enhanced_evaluators.py

import random
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)

def word_importance_analysis(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sentence: str) -> Dict[str, float]:
    words = sentence.split()
    importance_scores = {}

    for i, word in enumerate(words):
        masked_sentence = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
        inputs = tokenizer(masked_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        masked_token_logits = logits[0, i]
        original_token_id = tokenizer.encode(word, add_special_tokens=False)[0]
        importance_score = masked_token_logits[original_token_id].item()
        importance_scores[word] = importance_score

    return importance_scores

def create_contrast_set(sentence: str, num_contrasts: int = 5) -> List[str]:
    words = sentence.split()
    contrast_set = []

    for _ in range(num_contrasts):
        word_index = random.randint(0, len(words) - 1)
        original_word = words[word_index]
        
        synsets = wordnet.synsets(original_word)
        if synsets:
            if random.choice([True, False]):  # Randomly choose synonym or antonym
                new_word = random.choice(synsets).lemmas()[0].name()
            else:
                antonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        antonyms.extend(lemma.antonyms())
                if antonyms:
                    new_word = random.choice(antonyms).name()
                else:
                    new_word = original_word
        else:
            new_word = original_word

        new_sentence = ' '.join(words[:word_index] + [new_word] + words[word_index+1:])
        contrast_set.append(new_sentence)

    return contrast_set

def bias_detection(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sentence: str, target_words: List[str]) -> Dict[str, float]:
    bias_scores = {}

    for word in target_words:
        original_sentence = sentence.replace("[TARGET]", word)
        inputs = tokenizer(original_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        word_id = tokenizer.encode(word, add_special_tokens=False)[0]
        bias_score = logits[0, -1, word_id].item()
        bias_scores[word] = bias_score

    return bias_scores

def run_enhanced_evaluation(query: str, response: str, model_name: str = "distilgpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    importance_scores = word_importance_analysis(model, tokenizer, query)
    contrast_set = create_contrast_set(query)
    target_words = ["laptop", "smartphone", "book"]  # Example target words
    bias_scores = bias_detection(model, tokenizer, "I want to return the [TARGET] I bought.", target_words)

    return {
        "word_importance": importance_scores,
        "contrast_set": contrast_set,
        "bias_scores": bias_scores
    }

if __name__ == "__main__":
    # Example usage
    query = "I want to return the laptop I bought last month. What's the process?"
    response = "Thank you for contacting us about returning your laptop. Here are the steps to return your product: 1) Pack the laptop in its original packaging. 2) Include all accessories. 3) Print the return label from your account. 4) Drop off the package at any authorized shipping location. Once we receive the item, we'll process your refund within 5-7 business days."

    results = run_enhanced_evaluation(query, response)

    print("Word Importance Analysis:")
    for word, score in results["word_importance"].items():
        print(f"  {word}: {score:.4f}")

    print("\nContrast Set:")
    for i, contrast in enumerate(results["contrast_set"], 1):
        print(f"  Contrast {i}: {contrast}")

    print("\nBias Detection:")
    for word, score in results["bias_scores"].items():
        print(f"  {word}: {score:.4f}")