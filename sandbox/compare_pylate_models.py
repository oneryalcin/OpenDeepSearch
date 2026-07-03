#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from opendeepsearch.ranking_models.pylate_reranker import PyLateReranker
import time

def test_model(model_name, documents, queries):
    """Test a specific PyLate model and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        start = time.time()
        reranker = PyLateReranker(
            model_name=model_name,
            device="cpu"
        )
        init_time = time.time() - start
        print(f"Initialization time: {init_time:.2f}s")
        
        # Test with a programming query
        query = queries[0]
        scores = reranker.calculate_scores([query], documents, normalize=None)
        scores_list = scores[0].tolist()
        
        # Calculate score statistics
        min_score = min(scores_list)
        max_score = max(scores_list)
        score_range = max_score - min_score
        
        # Find best and worst matches
        sorted_indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        
        print(f"\nQuery: '{query}'")
        print(f"Score range: {min_score:.2f} to {max_score:.2f} (range: {score_range:.2f})")
        print(f"\nTop 3 matches:")
        for i, idx in enumerate(sorted_indices[:3]):
            print(f"{i+1}. [Score: {scores_list[idx]:.2f}] {documents[idx][:70]}...")
        
        print(f"\nBottom 3 matches:")
        for i, idx in enumerate(sorted_indices[-3:]):
            print(f"{i+1}. [Score: {scores_list[idx]:.2f}] {documents[idx][:70]}...")
            
        return {
            'model': model_name,
            'init_time': init_time,
            'score_range': score_range,
            'min_score': min_score,
            'max_score': max_score,
            'scores': scores_list
        }
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return None

def main():
    # Documents with varying relevance
    documents = [
        # Highly relevant (programming)
        "Python is an interpreted programming language known for readability and simple syntax.",
        "JavaScript enables interactive web pages and is essential for front-end development.",
        "Learn programming by starting with basic concepts like variables and loops.",
        
        # Somewhat relevant (tech)
        "Artificial intelligence uses algorithms to simulate human intelligence.",
        "Database systems store and organize data for efficient retrieval.",
        
        # Completely irrelevant
        "Dolphins are marine mammals known for their intelligence.",
        "The Great Pyramid of Giza was built around 2560 BCE.",
        "Coffee beans are actually seeds from coffee plant berries.",
        "Beethoven composed nine symphonies during his lifetime.",
        "Antarctica is the coldest continent on Earth."
    ]
    
    queries = [
        "How to start learning programming for beginners",
        "Best practices for web development",
        "Database design principles"
    ]
    
    # PyLate models to test
    models = [
        "lightonai/answerai-colbert-small-v1",     # 33M params
        "lightonai/colbertv2.0",                   # 110M params
        "lightonai/GTE-ModernColBERT-v1",          # 149M params
        "lightonai/Reason-ModernColBERT",          # 149M params
    ]
    
    results = []
    for model in models:
        result = test_model(model, documents, queries)
        if result:
            results.append(result)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Init(s)':<8} {'Range':<8} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for r in results:
        model_short = r['model'].split('/')[-1][:40]
        print(f"{model_short:<45} {r['init_time']:<8.2f} {r['score_range']:<8.2f} {r['min_score']:<8.2f} {r['max_score']:<8.2f}")
    
    print("\nKey Insights:")
    print("- All ColBERT models show compressed score ranges due to MaxSim")
    print("- Even 'irrelevant' documents get relatively high scores")
    print("- Score differences of 0.5-1.0 are actually significant")
    print("- Larger models don't necessarily have better score separation")

if __name__ == "__main__":
    main()