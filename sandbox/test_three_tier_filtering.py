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

def test_three_tier_filtering():
    print("=" * 80)
    print("THREE-TIER FILTERING PIPELINE TEST")
    print("=" * 80)
    
    # Sample chunks (mix of relevant and irrelevant content)
    chunks = [
        # Highly relevant to Python programming
        "Python's syntax is designed for readability, using indentation to define code blocks instead of curly braces.",
        "The Python Package Index (PyPI) hosts over 400,000 third-party packages for various applications.",
        "Python supports multiple programming paradigms including object-oriented, functional, and procedural styles.",
        
        # Somewhat relevant (general programming)
        "JavaScript is primarily used for web development and runs in browsers and Node.js environments.",
        "Git version control helps developers track changes and collaborate on code projects.",
        "SQL databases store structured data in tables with defined relationships.",
        
        # Educational but irrelevant
        "The water cycle involves evaporation, condensation, and precipitation processes.",
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        "The Roman Empire lasted from 27 BC to 476 AD in the Western regions.",
        
        # Low quality / marketing
        "Click here for the best deals! Subscribe now!",
        "Download our app today! Special offer ends soon!",
        "Buy now and save 50%! Limited time only!",
        
        # Navigation / UI elements
        "Home | About | Contact | Services",
        "Previous Page | Next Page",
        "Copyright 2024. All rights reserved.",
        
        # More relevant Python content
        "List comprehensions in Python provide a concise way to create lists from existing iterables.",
        "Python's asyncio library enables asynchronous programming with async/await syntax.",
        "The Global Interpreter Lock (GIL) prevents multiple threads from executing Python bytecode simultaneously.",
        
        # Random low-quality
        "Read more...",
        "Loading...",
        "Please wait..."
    ]
    
    query = "How to learn Python programming effectively?"
    
    print(f"\nQuery: '{query}'")
    print(f"Total chunks: {len(chunks)}")
    
    # Initialize models
    print("\n1. INITIALIZING MODELS...")
    print("-" * 60)
    
    # Potion model
    print("Loading Potion-32M model...")
    start = time.time()
    potion_model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    potion_load_time = time.time() - start
    print(f"  Potion load time: {potion_load_time:.2f}s")
    
    # ColBERT model
    print("Loading ColBERT model...")
    start = time.time()
    colbert_reranker = PyLateReranker(model_name="lightonai/Reason-ModernColBERT", device="cpu")
    colbert_load_time = time.time() - start
    print(f"  ColBERT load time: {colbert_load_time:.2f}s")
    
    # TIER 1: FastText Quality Filtering
    print("\n2. TIER 1: FASTTEXT QUALITY FILTERING")
    print("-" * 60)
    
    start = time.time()
    quality_scores = predict_educational_value(chunks)
    fasttext_time = (time.time() - start) * 1000
    
    # Try different thresholds to find optimal cutoff
    quality_threshold = 0.1  # Lower threshold to catch only true noise
    tier1_chunks = [(chunk, i) for i, (chunk, score) in enumerate(zip(chunks, quality_scores)) if score > quality_threshold]
    
    print(f"Time: {fasttext_time:.1f}ms")
    print(f"Quality score distribution:")
    print(f"  > 1.0: {sum(1 for s in quality_scores if s > 1.0)} chunks")
    print(f"  0.5-1.0: {sum(1 for s in quality_scores if 0.5 <= s <= 1.0)} chunks")
    print(f"  0.1-0.5: {sum(1 for s in quality_scores if 0.1 <= s < 0.5)} chunks")
    print(f"  < 0.1: {sum(1 for s in quality_scores if s < 0.1)} chunks")
    print(f"\nUsing threshold: {quality_threshold}")
    print(f"Filtered: {len(chunks)} → {len(tier1_chunks)} chunks (removed {len(chunks)-len(tier1_chunks)})")
    print(f"\nRemoved chunks (score < {quality_threshold}):")
    for i, (chunk, score) in enumerate(zip(chunks, quality_scores)):
        if score <= quality_threshold:
            print(f"  [{score:.3f}] {chunk[:60]}...")
    print(f"\nKept chunks:")
    for chunk, orig_idx in tier1_chunks:
        score = quality_scores[orig_idx]
        print(f"  [{score:.3f}] {chunk[:60]}...")
    
    # TIER 2: Potion Semantic Filtering
    print("\n3. TIER 2: POTION SEMANTIC FILTERING")
    print("-" * 60)
    
    if tier1_chunks:
        # Encode query
        start = time.time()
        query_emb = potion_model.encode([query])
        query_time = (time.time() - start) * 1000
        
        # Encode chunks
        start = time.time()
        chunk_texts = [chunk for chunk, _ in tier1_chunks]
        chunk_embs = potion_model.encode(chunk_texts)
        chunk_time = (time.time() - start) * 1000
        
        # Calculate similarities
        start = time.time()
        similarities = cosine_similarity(query_emb, chunk_embs).flatten()
        sim_time = (time.time() - start) * 1000
        
        potion_time = query_time + chunk_time + sim_time
        
        # Filter by semantic similarity
        semantic_threshold = 0.3  # Lower threshold for static embeddings
        tier2_chunks = [(chunk, orig_idx, sim) for (chunk, orig_idx), sim in zip(tier1_chunks, similarities) if sim > semantic_threshold]
        tier2_chunks.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Time: {potion_time:.1f}ms (query: {query_time:.1f}ms, chunks: {chunk_time:.1f}ms, sim: {sim_time:.1f}ms)")
        print(f"Filtered: {len(tier1_chunks)} → {len(tier2_chunks)} chunks")
        print(f"Semantic scores:")
        for chunk, orig_idx, sim in tier2_chunks:
            print(f"  [{sim:.3f}] {chunk[:60]}...")
    
    # TIER 3: ColBERT Fine Reranking
    print("\n4. TIER 3: COLBERT FINE RERANKING")
    print("-" * 60)
    
    if tier2_chunks:
        start = time.time()
        final_chunks = [chunk for chunk, _, _ in tier2_chunks]
        colbert_scores = colbert_reranker.calculate_scores([query], final_chunks, normalize=None)
        colbert_time = (time.time() - start) * 1000
        
        # Get final ranking
        scores_array = colbert_scores[0].numpy()
        ranked_indices = np.argsort(scores_array)[::-1]
        
        print(f"Time: {colbert_time:.1f}ms")
        print(f"Final ranking (top 10):")
        for i, idx in enumerate(ranked_indices[:10]):
            chunk, orig_idx, potion_score = tier2_chunks[idx]
            colbert_score = scores_array[idx]
            print(f"  {i+1}. [ColBERT: {colbert_score:.2f}, Potion: {potion_score:.3f}] {chunk[:60]}...")
    
    # COMPARISON
    print("\n5. PIPELINE COMPARISON")
    print("-" * 60)
    
    # Baseline: ColBERT only
    print("Baseline (ColBERT on all chunks):")
    start = time.time()
    baseline_scores = colbert_reranker.calculate_scores([query], chunks, normalize=None)
    baseline_time = (time.time() - start) * 1000
    print(f"  Time: {baseline_time:.1f}ms for {len(chunks)} chunks")
    
    # Three-tier total
    total_time = fasttext_time + potion_time + colbert_time
    print(f"\nThree-tier pipeline:")
    print(f"  FastText:  {fasttext_time:6.1f}ms ({len(chunks)} → {len(tier1_chunks)} chunks)")
    print(f"  Potion:    {potion_time:6.1f}ms ({len(tier1_chunks)} → {len(tier2_chunks)} chunks)")
    print(f"  ColBERT:   {colbert_time:6.1f}ms ({len(tier2_chunks)} chunks)")
    print(f"  Total:     {total_time:6.1f}ms")
    print(f"\nSpeedup: {baseline_time/total_time:.1f}x faster!")
    print(f"Chunks processed by ColBERT: {len(tier2_chunks)}/{len(chunks)} ({len(tier2_chunks)/len(chunks)*100:.0f}%)")

if __name__ == "__main__":
    test_three_tier_filtering()
