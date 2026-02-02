import torch
import numpy as np
import pandas as pd
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
# import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from .mean_pooling import mean_pooling

from .config_loader import load_config
CONFIG_PATH = Path.cwd() / "config.json"
cfg = load_config(CONFIG_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_embeddings(model_id, texts, batch_size):
    """
    Universal function to compute embeddings.
    Automatically detects if model is compatible with SentenceTransformer,
    otherwise falls back to Hugging Face AutoModel.
    
    Args:
        model_id (str): The model identifier.
        texts (list): List of input text strings.
        batch_size (int): Batch size for inference.
        
    Returns: numpy array of embeddings
    """
    print(f"Computing embeddings for {model_id}...")

    # 1. Try SentenceTransformer
    try:
        print(f"Attempting to load {model_id} as SentenceTransformer...")
        model = SentenceTransformer(model_id, device=str(DEVICE))
        
        # If we get here, model loaded successfully
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

        # enforce L1 or L2-normalization
        norm_p = cfg['normalization']
        if norm_p > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True, ord=norm_p)
            norms = np.clip(norms, 1e-9, None)
            embeddings = embeddings / norms

        # cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("Success with SentenceTransformer.")
        return embeddings

    except Exception as e:
        print(f"SentenceTransformer load/run failed ({e}). Falling back to AutoModel.")

        # 2. Fallback to AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # set a reasonable max length to avoid truncation warning
        if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 1e6:
            tokenizer.model_max_length = config.get('tokenizer_model_max_length', 128)

        model = AutoModel.from_pretrained(model_id)
        model.to(DEVICE)
        model.eval()

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"AutoModel {model_id}", leave=False):
                batch_texts = texts[i:i+batch_size]
                enc = tokenizer(batch_texts, padding=True, truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                last_hidden = outputs.last_hidden_state
                pooled = mean_pooling(last_hidden, attention_mask)
                normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.extend(normed.cpu().numpy())

        # cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return np.vstack(embeddings)
