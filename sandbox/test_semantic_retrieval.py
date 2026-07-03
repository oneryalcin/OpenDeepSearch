#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from opendeepsearch.context_scraping.utils import predict_educational_value
from opendeepsearch.ranking_models.pylate_reranker import PyLateReranker
from model2vec import StaticModel
import numpy as np
import time

def cosine_similarity(a, b):
    """Calculate cosine similarity between vectors"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def test_semantic_retrieval():
    print("=" * 80)
    print("SEMANTIC RETRIEVAL TEST - NO KEYWORD OVERLAP")
    print("=" * 80)
    
    # All chunks pooled together - each question gets ALL chunks
    all_chunks = [
        # Scalability/Performance chunks
        "Horizontal scaling involves adding more machines to a resource pool, while vertical scaling means upgrading existing hardware. Load balancers distribute incoming requests across multiple servers.",
        "Memory access patterns significantly impact performance. Cache misses force the CPU to fetch data from slower main memory. Algorithms with poor locality of reference suffer from frequent cache misses.",
        "Systems have finite resources like memory, CPU, and network bandwidth. When demand exceeds capacity, resource exhaustion occurs. Connection pools, rate limiting, and auto-scaling help manage resource consumption.",
        
        # Security chunks
        "Cryptographic hashing produces a fixed-size output from arbitrary input. Salt adds randomness before hashing to prevent rainbow table attacks. Modern algorithms like bcrypt and argon2 include built-in salting and stretching.",
        "Encryption transforms data into a form that cannot be easily understood by unauthorized people.",
        
        # Testing/Quality chunks
        "Unit tests verify individual components in isolation. Integration tests check interactions between components. Regression tests catch bugs introduced by modifications. Continuous integration automatically runs test suites.",
        "Version control systems track changes to code over time, enabling collaboration and history management.",
        
        # Architecture/Design chunks
        "Microservices architecture decomposes applications into small, independent services that communicate via APIs.",
        "Object-oriented programming organizes code into classes and objects, promoting code reuse and modularity.",
        "Functional programming treats computation as evaluation of mathematical functions and avoids changing state.",
        "Dependency injection decouples components by providing dependencies from external sources rather than creating them internally.",
        
        # Web/Network chunks
        "The TCP/IP protocol stack consists of four layers: application, transport, internet, and link layer.",
        "RESTful APIs use HTTP methods like GET, POST, PUT, and DELETE to perform operations on resources.",
        "WebSockets enable full-duplex communication channels over a single TCP connection.",
        "CSS Grid Layout provides a two-dimensional layout system, making it easier to design complex responsive layouts.",
        
        # Database chunks
        "Database indexing improves query performance by creating data structures that allow faster lookups.",
        "SQL joins combine rows from two or more tables based on related columns between them.",
        "NoSQL databases provide flexible schemas and horizontal scalability for unstructured data.",
        
        # Algorithm/Data Structure chunks
        "Binary search trees maintain sorted data and allow searches, insertions, and deletions in logarithmic time.",
        "Dynamic programming optimizes recursive algorithms by storing results of subproblems to avoid redundant calculations.",
        "Recursion solves problems by breaking them into smaller subproblems of the same type.",
        
        # Misc tech chunks
        "Machine learning models learn patterns from training data to make predictions on new inputs.",
        "Blockchain technology creates an immutable ledger of transactions using cryptographic hashing.",
        "Git branches allow developers to work on features independently without affecting the main codebase.",
        "API rate limiting prevents abuse by restricting the number of requests a client can make in a given timeframe.",
        
        # Low quality/noise chunks
        "Click here to learn more about our services.",
        "Subscribe to our newsletter for updates.",
        "Copyright 2024. All rights reserved.",
        "Loading... Please wait.",
        "Page not found. Return to home.",
    ]
    
    # Test cases - all use the same chunk pool
    test_cases = [
        {
            "query": "How to make a web application handle thousands of users simultaneously?",
            "gold_answer_idx": 0,  # "Horizontal scaling..." chunk
        },
        {
            "query": "What causes programs to run slowly when processing large datasets?",
            "gold_answer_idx": 1,  # "Memory access patterns..." chunk
        },
        {
            "query": "How do I ensure my code keeps working correctly after making changes?",
            "gold_answer_idx": 5,  # "Unit tests..." chunk
        },
        {
            "query": "Why does my server crash under heavy traffic?",
            "gold_answer_idx": 2,  # "Systems have finite resources..." chunk
        },
        {
            "query": "How to store user passwords securely in a database?",
            "gold_answer_idx": 3,  # "Cryptographic hashing..." chunk
        }
    ]
    
    # Initialize models
    print("\nInitializing models...")
    potion_model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    colbert_reranker = PyLateReranker(model_name="lightonai/colbertv2.0", device="cpu")
    
    # Test each case
    results = {"fasttext": [], "potion": [], "colbert": []}
    
    print(f"\nTotal chunks in pool: {len(all_chunks)}")
    print(f"Educational chunks: {len(all_chunks) - 5}")
    print(f"Noise chunks: 5")
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*60}")
        print(f"Query: {test_case['query']}")
        print(f"Gold answer is at index: {test_case['gold_answer_idx']}")
        
        query = test_case['query']
        chunks = all_chunks  # Use the shared pool
        gold_idx = test_case['gold_answer_idx']
        
        # 1. FastText scoring (not for retrieval, just for filtering)
        quality_scores = predict_educational_value(chunks)
        
        # 2. Potion semantic search
        print("\nPOTION RETRIEVAL:")
        query_emb = potion_model.encode([query])
        chunk_embs = potion_model.encode(chunks)
        potion_sims = cosine_similarity(query_emb, chunk_embs).flatten()
        potion_ranking = np.argsort(potion_sims)[::-1]
        
        potion_rank = list(potion_ranking).index(gold_idx) + 1
        results["potion"].append(potion_rank)
        
        print(f"Gold answer ranked: #{potion_rank}")
        for j, idx in enumerate(potion_ranking[:3]):
            print(f"  {j+1}. [{potion_sims[idx]:.3f}] {chunks[idx][:80]}...")
        
        # 3. ColBERT reranking
        print("\nCOLBERT RETRIEVAL:")
        colbert_scores = colbert_reranker.calculate_scores([query], chunks, normalize=None)
        colbert_scores_array = colbert_scores[0].numpy()
        colbert_ranking = np.argsort(colbert_scores_array)[::-1]
        
        colbert_rank = list(colbert_ranking).index(gold_idx) + 1
        results["colbert"].append(colbert_rank)
        
        print(f"Gold answer ranked: #{colbert_rank}")
        for j, idx in enumerate(colbert_ranking[:3]):
            print(f"  {j+1}. [{colbert_scores_array[idx]:.2f}] {chunks[idx][:80]}...")
        
        # Show gold answer
        print(f"\nGOLD ANSWER:")
        print(f"  {chunks[gold_idx]}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - GOLD ANSWER RANKINGS")
    print(f"{'='*60}")
    print(f"{'Test Case':<15} {'Potion Rank':<15} {'ColBERT Rank':<15}")
    print("-" * 45)
    
    for i in range(len(test_cases)):
        print(f"Test {i+1:<10} {results['potion'][i]:<15} {results['colbert'][i]:<15}")
    
    print(f"\nAverage Rankings (lower is better):")
    print(f"  Potion:  {np.mean(results['potion']):.1f}")
    print(f"  ColBERT: {np.mean(results['colbert']):.1f}")
    
    print(f"\nTop-1 Accuracy:")
    print(f"  Potion:  {sum(1 for r in results['potion'] if r == 1)}/{len(results['potion'])}")
    print(f"  ColBERT: {sum(1 for r in results['colbert'] if r == 1)}/{len(results['colbert'])}")

if __name__ == "__main__":
    test_semantic_retrieval()