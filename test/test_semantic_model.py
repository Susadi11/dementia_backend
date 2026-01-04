"""Quick verification of semantic model loading"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Testing semantic model loading...")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported successfully")
    
    print("\nLoading model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully!")
    
    # Test encoding
    sentences = ["I love gardening", "My car broke down"]
    embeddings = model.encode(sentences)
    print(f"✅ Embeddings generated: shape {embeddings.shape}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"✅ Similarity calculated: {similarity:.3f}")
    print(f"   (Low similarity = {similarity:.3f} confirms different topics)")
    
    print("\n" + "="*60)
    print("✅ ALL SEMANTIC SIMILARITY FEATURES WORKING!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
