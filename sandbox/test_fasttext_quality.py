#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from opendeepsearch.context_scraping.utils import predict_educational_value, clean_markdown_links, filter_quality_content
import time

def test_fasttext_quality_scoring():
    # Test samples of varying quality
    test_texts = {
        "High Quality - Technical": """
Machine learning models require careful consideration of the bias-variance tradeoff. 
High bias models tend to underfit the data, missing important patterns and relationships. 
Conversely, high variance models overfit to the training data, capturing noise rather than 
true underlying patterns. The optimal model balances these competing concerns through 
techniques like regularization, cross-validation, and ensemble methods.
        """,
        
        "High Quality - Educational": """
The Renaissance period marked a profound transformation in European culture, spanning roughly 
from the 14th to 17th centuries. This era witnessed unprecedented advances in art, science, 
and philosophy. Leonardo da Vinci exemplified the Renaissance ideal of the polymath, contributing 
to fields as diverse as anatomy, engineering, and painting. His systematic observations and 
detailed sketches laid groundwork for modern scientific illustration.
        """,
        
        "Medium Quality - Blog Post": """
Today I want to talk about Python programming. Python is really popular these days and 
lots of people use it. You can do web development with Django or Flask. Data science 
is another area where Python shines. Libraries like pandas and numpy make it easy to 
work with data. Machine learning is also possible with scikit-learn and tensorflow.
        """,
        
        "Low Quality - Marketing": """
Buy now! Best deals! Click here for amazing offers! 
Subscribe to our newsletter. Follow us on social media.
Download our app today. Special promotion ends soon!
Contact us for more information. Terms and conditions apply.
        """,
        
        "Low Quality - Navigation": """
Home | About | Services | Contact
Previous Next
Share Trade More Buy Sell Download Menu
Currency: USD
You Buy: 100 BTC
You Receive: 4,250,000 USD
≈ 42,500 per BTC
        """,
        
        "Mixed Quality - With Code": """
Here's how to implement a binary search in Python:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

This algorithm has O(log n) time complexity, making it efficient for sorted arrays.
        """,
        
        "Low Quality - Short Fragments": """
Python
Learn more
Click here
See details
Read documentation
Try it now
Get started
        """,
        
        "High Quality - Scientific": """
Photosynthesis converts light energy into chemical energy through two main stages: 
light-dependent reactions and the Calvin cycle. During light reactions, chlorophyll 
molecules absorb photons, exciting electrons to higher energy states. These electrons 
flow through the electron transport chain, generating ATP and NADPH. The Calvin cycle 
then uses these energy carriers to fix carbon dioxide into glucose molecules.
        """,
        
        "Medium Quality - Tutorial": """
To get started with React, you'll need Node.js installed on your computer. 
First, create a new app using create-react-app. Then you can start building 
components. Components are the building blocks of React applications. Each 
component can have its own state and props. State is internal data that can 
change, while props are passed from parent components.
        """,
        
        "Low Quality - Repetitive": """
Best price best price best price. Quality service quality service.
Fast delivery fast delivery. Customer satisfaction guaranteed.
Order now order now. Limited time offer limited time.
Best deals best deals. Shop now shop now.
        """
    }
    
    print("=" * 80)
    print("FASTTEXT QUALITY SCORING TEST")
    print("=" * 80)
    
    # Test raw scoring
    print("\n1. RAW SCORING (0-2 scale)")
    print("-" * 60)
    
    scores = []
    for name, text in test_texts.items():
        start = time.time()
        score = predict_educational_value([text])[0]
        elapsed = (time.time() - start) * 1000
        scores.append((name, score, elapsed))
        print(f"{name:<35} Score: {score:.3f}  Time: {elapsed:.1f}ms")
    
    # Sort by score
    print("\n2. RANKED BY QUALITY SCORE")
    print("-" * 60)
    scores.sort(key=lambda x: x[1], reverse=True)
    for name, score, _ in scores:
        bar = "█" * int(score * 20)
        print(f"{score:.3f} {bar:<40} {name}")
    
    # Test filtering function
    print("\n3. TESTING filter_quality_content() with threshold=0.5")
    print("-" * 60)
    
    mixed_content = """
Buy now! Best deals ever! Click here!
Subscribe to our newsletter today!

Machine learning revolutionized how we process and understand data. 
Neural networks, inspired by biological neurons, can learn complex patterns 
through backpropagation. Deep learning extends this with multiple hidden layers,
enabling breakthrough performance in computer vision and natural language processing.

Download our app! Special offer!
Follow us on social media!

The scientific method involves systematic observation, measurement, and experiment.
Hypotheses must be testable and falsifiable. Through rigorous testing and peer review,
scientific knowledge advances incrementally, building upon previous discoveries.

Contact us | Terms | Privacy
Home > Products > Details
    """
    
    print("Original content length:", len(mixed_content))
    filtered = filter_quality_content(mixed_content, min_quality_score=0.5)
    print("Filtered content length:", len(filtered))
    print("\nFiltered content:")
    print("-" * 40)
    print(filtered)
    
    # Test speed with batches
    print("\n4. BATCH PROCESSING SPEED TEST")
    print("-" * 60)
    
    batch_sizes = [1, 10, 50, 100]
    sample_text = test_texts["Medium Quality - Blog Post"]
    
    for batch_size in batch_sizes:
        texts = [sample_text] * batch_size
        start = time.time()
        scores = predict_educational_value(texts)
        elapsed = (time.time() - start) * 1000
        per_text = elapsed / batch_size
        print(f"Batch size {batch_size:3d}: {elapsed:6.1f}ms total, {per_text:5.2f}ms per text")

if __name__ == "__main__":
    test_fasttext_quality_scoring()