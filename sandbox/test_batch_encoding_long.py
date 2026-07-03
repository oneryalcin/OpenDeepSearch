#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from opendeepsearch.ranking_models.pylate_reranker import PyLateReranker
import time
import numpy as np
from transformers import AutoTokenizer

def test_batch_encoding_with_long_docs():
    # Long document (300+ tokens)
    long_doc = """Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions. Python 2.7.18, released in 2020, was the last release of Python 2.

Python consistently ranks as one of the most popular programming languages, and has gained widespread use in the machine learning community. It has been widely adopted in artificial intelligence research, data science, web development, scientific computing, and many other domains. Major organizations like Google, NASA, Netflix, Instagram, and Spotify use Python extensively in their technology stacks.

The language's simplicity and readability make it an excellent choice for beginners, while its powerful features and extensive ecosystem of third-party packages make it suitable for complex, large-scale applications. Python's philosophy is summarized in The Zen of Python, a collection of aphorisms that capture the language's design principles. These include "Beautiful is better than ugly," "Explicit is better than implicit," and "Simple is better than complex."

Python's standard library is extensive, offering modules and packages for tasks ranging from web services to operating system interfaces. The Python Package Index (PyPI) hosts hundreds of thousands of third-party packages, extending Python's capabilities to virtually every domain of computing. Popular frameworks include Django and Flask for web development, NumPy and Pandas for data analysis, TensorFlow and PyTorch for machine learning, and many more specialized tools."""
    
    # Check token count
    tokenizer = AutoTokenizer.from_pretrained('lightonai/answerai-colbert-small-v1')
    tokens = tokenizer.encode(long_doc)
    print(f"Document token count: {len(tokens)} tokens")
    
    # Initialize reranker
    print("\nInitializing PyLate reranker...")
    reranker = PyLateReranker(
        model_name="lightonai/answerai-colbert-small-v1",
        device="cpu"
    )
    
    # Test different batch sizes with long documents
    batch_sizes = [1, 10, 25, 50, 100]
    
    print(f"\n{'='*70}")
    print("BATCH ENCODING PERFORMANCE WITH LONG DOCUMENTS (417 tokens)")
    print(f"{'='*70}")
    print(f"{'Batch Size':<12} {'Total Time (ms)':<20} {'Per Doc (ms)':<15} {'Speedup':<10}")
    print("-" * 65)
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch of long documents
        documents = [long_doc] * batch_size
        
        # Warm up
        _ = reranker.model.encode(["warm up"], is_query=False)
        
        # Time the encoding (multiple runs for average)
        times = []
        for _ in range(5 if batch_size <= 50 else 3):  # Fewer runs for large batches
            start = time.time()
            embeddings = reranker.model.encode(documents, is_query=False)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        per_doc = avg_time / batch_size
        speedup = results.get(1, {}).get('per_doc', per_doc) / per_doc if 1 in results else 1.0
        
        results[batch_size] = {
            'total': avg_time,
            'per_doc': per_doc,
            'speedup': speedup
        }
        
        print(f"{batch_size:<12} {avg_time:<20.1f} {per_doc:<15.2f} {speedup:<10.1f}x")
    
    # Test realistic scenario: reranking 100 documents
    print(f"\n{'='*70}")
    print("REALISTIC RERANKING SCENARIO")
    print(f"{'='*70}")
    
    query = "What are the key features and philosophy of Python programming language?"
    documents_100 = [long_doc] * 100
    
    # Time full reranking process
    print("\nReranking 100 long documents (417 tokens each)...")
    
    # Breakdown timing
    start_total = time.time()
    
    # Query encoding
    start = time.time()
    query_emb = reranker.model.encode([query], is_query=True)
    query_time = (time.time() - start) * 1000
    
    # Document encoding (batch)
    start = time.time()
    doc_emb = reranker.model.encode(documents_100, is_query=False)
    doc_time = (time.time() - start) * 1000
    
    # Similarity calculation
    start = time.time()
    scores = reranker.model.similarity(query_emb, doc_emb)
    sim_time = (time.time() - start) * 1000
    
    total_time = (time.time() - start_total) * 1000
    
    print(f"\nTiming breakdown:")
    print(f"  Query encoding (1 query):     {query_time:.1f}ms")
    print(f"  Document encoding (100 docs): {doc_time:.1f}ms ({doc_time/100:.2f}ms per doc)")
    print(f"  Similarity calculation:       {sim_time:.1f}ms")
    print(f"  Total reranking time:         {total_time:.1f}ms")
    
    # Compare with serial encoding
    serial_time = results[1]['per_doc'] * 100 + query_time + sim_time
    print(f"\nComparison:")
    print(f"  Serial encoding (1-by-1):     {serial_time:.1f}ms")
    print(f"  Batch encoding:               {total_time:.1f}ms")
    print(f"  Speedup:                      {serial_time/total_time:.1f}x")

if __name__ == "__main__":
    test_batch_encoding_with_long_docs()