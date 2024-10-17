# openai_enhanced_evaluators.py

from openai import OpenAI

client = OpenAI()
import random
from typing import List, Dict
import json

# Make sure to set your OpenAI API key
# openai.api_key = "your-api-key-here"

from openai import OpenAI

client = OpenAI()

def word_importance_analysis(sentence: str) -> Dict[str, float]:
    words = sentence.split()
    importance_scores = {}

    for i, word in enumerate(words):
        masked_sentence = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
        
        functions = [
            {
                "name": "get_word_importance",
                "description": "Get the importance score of a masked word in a sentence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "importance_score": {
                            "type": "number",
                            "description": "The importance score of the masked word, from 0 to 1"
                        }
                    },
                    "required": ["importance_score"]
                }
            }
        ]

        messages = [
            {"role": "system", "content": "You are an AI assistant that analyzes word importance in sentences."},
            {"role": "user", "content": f"Original sentence: {sentence}\nMasked sentence: {masked_sentence}\n\nHow important is the masked word for understanding the sentence? Rate from 0 to 1."}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4",  # or "gpt-4o-mini" if available
                messages=messages,
                functions=functions,
                function_call={"name": "get_word_importance"}
            )
            function_args = response.choices[0].message.function_call.arguments
            importance_score = eval(function_args)['importance_score']
            importance_scores[word] = importance_score
            print(f"Word '{word}' importance score: {importance_score}")

        except Exception as e:
            print(f"Error processing word '{word}': {str(e)}")
            importance_scores[word] = None

    return importance_scores

def create_contrast_set(sentence: str, num_contrasts: int = 5) -> List[str]:
    functions = [
        {
            "name": "generate_contrast_set",
            "description": "Generate a set of contrasting sentences",
            "parameters": {
                "type": "object",
                "properties": {
                    "contrasts": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of contrasting sentences"
                    }
                },
                "required": ["contrasts"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are an AI assistant that generates contrasting sentences."},
        {"role": "user", "content": f"Generate {num_contrasts} variations of the following sentence, each with a small but meaningful change that could alter its interpretation:\n\n{sentence}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=functions,
            function_call={"name": "generate_contrast_set"}
        )

        function_args = json.loads(response.choices[0].message.function_call.arguments)
        contrast_set = function_args['contrasts']
        return contrast_set[:num_contrasts]

    except Exception as e:
        print(f"Error generating contrast set: {str(e)}")
        return []

import re

def bias_detection2(sentence: str, target_words: List[str]) -> Dict[str, float]:
    functions = [
        {
            "name": "detect_bias",
            "description": "Detect bias in sentences with different target words",
            "parameters": {
                "type": "object",
                "properties": {
                    "bias_scores": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "number"
                        },
                        "description": "Bias scores for each target word"
                    }
                },
                "required": ["bias_scores"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are an AI assistant that detects bias in sentences."},
        {"role": "user", "content": f"Evaluate the potential bias in the following sentence when [TARGET] is replaced with each of these words: {', '.join(target_words)}. Provide a bias score from 0 (no bias) to 1 (strong bias) for each word.\n\nTemplate sentence: {sentence}\n\nReturn the results as a JSON object with the words as keys and scores as values."}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=functions,
            function_call={"name": "detect_bias"}
        )
        print(f'response: {response}')
        function_args = json.loads(response.choices[0].message.function_call.arguments)
        bias_scores = function_args['bias_scores']

        return bias_scores

    except Exception as e:
        print(f"Error detecting bias: {str(e)}")
        return {word: None for word in target_words}

def bias_detection(sentence: str, target_words: List[str]) -> Dict[str, float]:
    messages = [
        {"role": "system", "content": "You are an AI assistant that detects bias in sentences."},
        {"role": "user", "content": f"Evaluate the potential bias in the following sentence when [TARGET] is replaced with each of these words: {', '.join(target_words)}. Provide a bias score from 0 (no bias) to 1 (strong bias) for each word.\n\nTemplate sentence: {sentence}\n\nReturn the results in the following format:\nword1: score1\nword2: score2\n..."}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        print(f'response: {response}')
        content = response.choices[0].message.content
        bias_scores = {}
        for line in content.split('\n'):
            if ':' in line:
                word, score_str = line.split(':')
                word = word.strip()
                score = float(score_str.strip())
                if word in target_words:
                    bias_scores[word] = score

        return bias_scores

    except Exception as e:
        print(f"Error detecting bias: {str(e)}")
        return {word: None for word in target_words}

def run_enhanced_evaluation(query: str, response: str):
    importance_scores = word_importance_analysis(query)
    contrast_set = create_contrast_set(query)
    target_words = ["laptop", "smartphone", "book"]  # Example target words
    bias_scores = bias_detection("I want to return the [TARGET] I bought.", target_words)

    return {
        "word_importance": importance_scores,
        "contrast_set": contrast_set,
        "bias_scores": bias_scores
    }
    
if __name__ == "__main__":
    # Example usage
    query = "I want to return the laptop I bought last month. What's the process?"
    response = "Thank you for contacting us about returning your laptop. Here are the steps to return your product: 1) Pack the laptop in its original packaging. 2) Include all accessories. 3) Print the return label from your account. 4) Drop off the package at any authorized shipping location. Once we receive the item, we'll process your refund within 5-7 business days."

    # results = run_enhanced_evaluation(query, response)

    # print("Word Importance Analysis:")
    # for word, score in results["word_importance"].items():
    #     print(f"  {word}: {score:.4f}")

    # print("\nContrast Set:")
    # for i, contrast in enumerate(results["contrast_set"], 1):
    #     print(f"  Contrast {i}: {contrast}")

    # print("\nBias Detection:")
    # for word, score in results["bias_scores"].items():
    #     print(f"  {word}: {score}")
        
    print("\n3. Bias Detection")
    template = "The [TARGET] employee was passed over for promotion despite having excellent qualifications."
    targets = ["male", "female", "older", "younger", "immigrant", "local"]
    bias_scores = bias_detection(template, targets)
    for word, score in bias_scores.items():
        print(f"  {word}: {score}")