#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from opendeepsearch.ranking_models.pylate_reranker import PyLateReranker
import time
import torch

def test_pylate_reranker():
    print("Testing PyLate Reranker...")
    
    # Test documents with semantic meaning but no direct keyword overlap
    documents = [
        # Geography/Travel (avoiding direct city/country names)
        "The City of Light features an iron lattice tower that has become a global cultural icon.",
        "This Asian megacity seamlessly blends cutting-edge technology with centuries-old shrines.",
        "The world's largest tropical wetland ecosystem produces 20% of Earth's oxygen.",
        
        # Technology/Programming (avoiding language names)
        "This interpreted language emphasizes code readability and uses indentation for block structure.",
        "Statistical pattern recognition algorithms improve performance through exposure to training examples.",
        "A declarative component-based library maintained by Meta for creating interactive UIs.",
        
        # Science/Medicine (avoiding technical terms in queries)
        "mRNA instructions teach immune cells to identify and destroy specific viral proteins.",
        "Green pigments in leaves capture solar radiation to produce glucose and release oxygen.",
        "The body's central processor consists of billions of interconnected electrochemical cells.",
        
        # Business/Finance (avoiding direct terms)
        "Global equity markets experienced severe downturns following subprime mortgage defaults.",
        "Digital currencies operate on distributed ledger systems without central authorities.",
        "Early-stage ventures pursue angel investors and institutional capital for growth.",
        
        # Food/Cooking (avoiding dish names)
        "Roman egg-based sauce combines aged pork jowl with hard cheese and black pepper.",
        "Vinegared rice paired with raw seafood became a culinary art form in Edo period.",
        "Butter folded repeatedly into yeasted dough creates flaky breakfast pastries.",
        
        # Sports/Fitness (avoiding sport names)
        "Endurance athletes gradually increase weekly mileage to prepare for 42.195 kilometer races.",
        "A PE teacher created an indoor winter activity using peach baskets in Massachusetts.",
        "Ancient Indian discipline harmonizes body movements with controlled breathing.",
        
        # History/Culture (avoiding period names)
        "European cultural rebirth from 1300-1600 marked renewed interest in classical antiquity.",
        "Massive stone structures along the Nile served as eternal resting places for rulers.",
        "Ancient defensive fortification spans thousands of li across northern territories.",
        
        # Entertainment/Arts (avoiding names)
        "Four Liverpool musicians transformed popular culture with experimental studio techniques.",
        "An enigmatic portrait by a Renaissance polymath captivates viewers with her subtle smile.",
        "A DVD-by-mail company disrupted traditional media distribution through internet delivery."
    ]
    
    queries = [
        # Technical queries (no keyword overlap)
        "Best language for beginners to start coding",
        "How does artificial intelligence learn from data",
        "Modern tools for building web applications",
        
        # Travel/Geography
        "Where is the Eiffel Tower located",
        "Largest rainforest on Earth",
        "Technology hub cities in Japan",
        
        # Health/Science
        "How do modern immunizations protect us",
        "Process plants use to make food from sunlight",
        "Number of nerve cells in human thinking organ",
        
        # Food/Cuisine
        "Authentic carbonara ingredients and method",
        "Traditional Japanese raw fish preparation",
        "How to make authentic croissants",
        
        # Finance/Business
        "Decentralized digital money systems",
        "Major economic collapse of late 2000s",
        "Raising capital for new companies",
        
        # Sports/Fitness
        "Preparing body for long-distance running events",
        "Origin story of basketball game",
        "Mind-body exercise from India",
        
        # History/Culture
        "European artistic revival period",
        "Tombs built by ancient Egyptians",
        "Longest man-made structure in China",
        
        # Arts/Entertainment
        "British band that changed rock music",
        "Most famous Da Vinci portrait",
        "Company that killed video rental stores"
    ]
    
    # Initialize reranker
    print("\nInitializing PyLate reranker...")
    start = time.time()
    reranker = PyLateReranker(
        model_name="lightonai/answerai-colbert-small-v1",  # Use the PyLate version
        device="cpu"  # Force CPU since MPS isn't supported
    )
    init_time = time.time() - start
    print(f"Initialization time: {init_time:.2f}s")
    
    # First, let's measure encoding time separately
    print(f"\n{'='*50}")
    print("ENCODING TIME BENCHMARKS")
    print(f"{'='*50}")
    
    # Test query encoding time
    print(f"\nEncoding {len(queries)} queries...")
    start = time.time()
    query_embeddings = reranker.model.encode(queries, is_query=True)
    query_encode_time = time.time() - start
    print(f"Query encoding time: {query_encode_time:.3f}s ({query_encode_time/len(queries)*1000:.1f}ms per query)")
    
    # Test document encoding time
    print(f"\nEncoding {len(documents)} documents...")
    start = time.time()
    doc_embeddings = reranker.model.encode(documents, is_query=False)
    doc_encode_time = time.time() - start
    print(f"Document encoding time: {doc_encode_time:.3f}s ({doc_encode_time/len(documents)*1000:.1f}ms per document)")
    
    # Test similarity calculation time
    print(f"\nCalculating similarity scores for {len(queries)}x{len(documents)} pairs...")
    start = time.time()
    similarity_scores = reranker.model.similarity(query_embeddings[:5], doc_embeddings)
    similarity_time = time.time() - start
    print(f"Similarity calculation time: {similarity_time:.3f}s")
    
    print(f"\n{'='*50}")
    print("RERANKING RESULTS (showing only first 5 queries)")
    print(f"{'='*50}")
    
    # Test reranking for first 5 queries to save output space
    for i, query in enumerate(queries[:5]):
        print(f"\n{'='*50}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*50}")
        
        # Breakdown timing
        start_total = time.time()
        
        # Query encoding
        start = time.time()
        q_emb = reranker.model.encode([query], is_query=True)
        q_time = time.time() - start
        
        # Document encoding (already cached in real scenario)
        start = time.time()
        d_emb = reranker.model.encode(documents, is_query=False)
        d_time = time.time() - start
        
        # Similarity calculation
        start = time.time()
        scores = reranker.model.similarity(q_emb, d_emb)
        sim_time = time.time() - start
        
        total_time = time.time() - start_total
        
        # Get top-k results
        scores_tensor = torch.tensor(scores[0])
        top_k = 3
        top_indices = torch.topk(scores_tensor, top_k).indices.tolist()
        
        print(f"\nTiming breakdown:")
        print(f"  - Query encoding: {q_time*1000:.1f}ms")
        print(f"  - Document encoding: {d_time*1000:.1f}ms")
        print(f"  - Similarity calculation: {sim_time*1000:.1f}ms")
        print(f"  - Total time: {total_time*1000:.1f}ms")
        
        print(f"\nTop {top_k} results:")
        for j, idx in enumerate(top_indices):
            print(f"{j+1}. [Score: {scores_tensor[idx]:.2f}] {documents[idx][:100]}...")
            
    # Summary statistics
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {reranker.model_name}")
    print(f"Device: {reranker.device}")
    print(f"Documents: {len(documents)}")
    print(f"Queries tested: {len(queries)}")
    print(f"Average query encoding: {query_encode_time/len(queries)*1000:.1f}ms")
    print(f"Average document encoding: {doc_encode_time/len(documents)*1000:.1f}ms")
    print(f"Total initialization time: {init_time:.2f}s")

if __name__ == "__main__":
    test_pylate_reranker()