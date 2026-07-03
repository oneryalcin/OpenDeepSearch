#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from opendeepsearch.ranking_models.pylate_reranker import PyLateReranker
import time
import numpy as np

def test_encoding_speed():
    models = [
        "lightonai/answerai-colbert-small-v1",
        "lightonai/colbertv2.0",
        "lightonai/GTE-ModernColBERT-v1",
        "lightonai/Reason-ModernColBERT",
    ]
    
    # Test texts of different lengths
    test_texts = {
        "short": ["How to learn Python?"] * 10,
        "medium": ["Python is a high-level programming language known for its simple syntax and readability. It's widely used in web development, data science, and automation."] * 10,
        "long": ["Python is a high-level, interpreted programming language known for its simple syntax and readability. It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected, supporting multiple programming paradigms including procedural, object-oriented, and functional programming."] * 10,
        "300_tokens": ["""Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions. Python 2.7.18, released in 2020, was the last release of Python 2.

Python consistently ranks as one of the most popular programming languages, and has gained widespread use in the machine learning community. It has been widely adopted in artificial intelligence research, data science, web development, scientific computing, and many other domains. Major organizations like Google, NASA, Netflix, Instagram, and Spotify use Python extensively in their technology stacks.

The language's simplicity and readability make it an excellent choice for beginners, while its powerful features and extensive ecosystem of third-party packages make it suitable for complex, large-scale applications. Python's philosophy is summarized in The Zen of Python, a collection of aphorisms that capture the language's design principles. These include "Beautiful is better than ugly," "Explicit is better than implicit," and "Simple is better than complex."

Python's standard library is extensive, offering modules and packages for tasks ranging from web services to operating system interfaces. The Python Package Index (PyPI) hosts hundreds of thousands of third-party packages, extending Python's capabilities to virtually every domain of computing. Popular frameworks include Django and Flask for web development, NumPy and Pandas for data analysis, TensorFlow and PyTorch for machine learning, and many more specialized tools."""] * 10
    }
    
    # Print token counts for reference
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('lightonai/answerai-colbert-small-v1')
    print("\nToken counts for test texts:")
    for name, texts in test_texts.items():
        tokens = tokenizer.encode(texts[0])
        print(f"  {name}: {len(tokens)} tokens")
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        # Initialize model
        reranker = PyLateReranker(model_name=model_name, device="cpu")
        model_short = model_name.split('/')[-1]
        results[model_short] = {}
        
        # Warm up
        _ = reranker.model.encode(["warm up"], is_query=True)
        
        for text_type, texts in test_texts.items():
            # Test query encoding
            times = []
            for _ in range(5):  # Multiple runs for average
                start = time.time()
                _ = reranker.model.encode(texts, is_query=True)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            per_text = avg_time / len(texts)
            results[model_short][f"query_{text_type}"] = per_text
            
            print(f"\nQuery encoding ({text_type} texts, {len(texts)} samples):")
            print(f"  Total: {avg_time:.1f}ms")
            print(f"  Per text: {per_text:.1f}ms")
            
            # Test document encoding
            times = []
            for _ in range(5):
                start = time.time()
                _ = reranker.model.encode(texts, is_query=False)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            per_text = avg_time / len(texts)
            results[model_short][f"doc_{text_type}"] = per_text
            
            print(f"Document encoding ({text_type} texts, {len(texts)} samples):")
            print(f"  Total: {avg_time:.1f}ms")
            print(f"  Per text: {per_text:.1f}ms")
    
    # Summary table
    print(f"\n{'='*100}")
    print("ENCODING SPEED COMPARISON (ms per text)")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Query':<30} {'Document':<30}")
    print(f"{'':<30} {'Short  Med   Long   300tok':<30} {'Short  Med   Long   300tok':<30}")
    print("-" * 90)
    
    for model in results:
        query_times = f"{results[model]['query_short']:5.1f} {results[model]['query_medium']:5.1f} {results[model]['query_long']:5.1f} {results[model]['query_300_tokens']:6.1f}"
        doc_times = f"{results[model]['doc_short']:5.1f} {results[model]['doc_medium']:5.1f} {results[model]['doc_long']:5.1f} {results[model]['doc_300_tokens']:6.1f}"
        print(f"{model:<30} {query_times:<30} {doc_times:<30}")
    
    # Test batch size impact
    print(f"\n{'='*60}")
    print("BATCH SIZE IMPACT (using answerai-colbert-small-v1)")
    print(f"{'='*60}")
    
    reranker = PyLateReranker(model_name="lightonai/answerai-colbert-small-v1", device="cpu")
    batch_sizes = [1, 5, 10, 25, 50, 100]
    
    for batch_size in batch_sizes:
        texts = ["Test sentence for batch processing."] * batch_size
        
        times = []
        for _ in range(5):
            start = time.time()
            _ = reranker.model.encode(texts, is_query=True)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        per_text = avg_time / batch_size
        
        print(f"Batch size {batch_size:3d}: {avg_time:6.1f}ms total, {per_text:5.2f}ms per text")

if __name__ == "__main__":
    test_encoding_speed()