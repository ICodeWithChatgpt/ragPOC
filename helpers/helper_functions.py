import json
import numpy as np


# Helper: Parse OpenAI Response
def parse_openai_response(raw_content):
    try:
        parsed_response = json.loads(raw_content)
        if not all(k in parsed_response for k in ["metadata", "tags", "summary", "normalized_version"]):
            print("Missing fields in OpenAI response.")
            return None
        return parsed_response
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
        return None


# Helper: Compute Cosine Similarity
def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Helper: truncate embeddings
def pad_or_truncate_embedding(embedding, target_length):
    """Pad or truncate the embedding to the target length."""
    if len(embedding) > target_length:
        return embedding[:target_length]
    elif len(embedding) < target_length:
        return np.pad(embedding, (0, target_length - len(embedding)), 'constant')
    return embedding