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
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
