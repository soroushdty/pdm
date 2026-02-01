import torch
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(last_hidden_state, attention_mask):
    """Averages token embeddings while ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def get_embedding(text, model_id, batch_size=32):
    """
    Receives text (str or list) and model name.
    Returns a normalized numpy array of embeddings.
    """
    # Ensure text is a list for batch processing
    texts = [text] if isinstance(text, str) else text
    
    # 1. Try SentenceTransformers first
    try:
        model = SentenceTransformer(model_id, device=str(DEVICE))
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return embeddings

    # 2. Fallback to AutoModel for generic Transformer models
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
        
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                
                outputs = model(**enc)
                pooled = mean_pooling(outputs.last_hidden_state, enc['attention_mask'])
                # L2 Normalization (Unit Vector)
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.extend(normed.cpu().numpy())

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        return np.vstack(all_embeddings)
