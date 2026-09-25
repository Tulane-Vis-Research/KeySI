"""Shared data, models, configuration, and preprocessing for KeySI."""



import os
import shutil
import re 
import json
import json as json_module
import random
import math
import gc
import logging
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
import traceback
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from rank_bm25 import BM25Okapi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, AutoTokenizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    pairwise_distances,
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    adjusted_rand_score, 
    normalized_mutual_info_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine


PROJECT_ROOT = Path(os.getenv("KEYSI_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
BASE_DIR = str(PROJECT_ROOT)
DATA_DIR = Path(os.getenv("KEYSI_DATA_DIR", PROJECT_ROOT / "CSV")).resolve()
OUTPUT_DIR = str(Path(os.getenv("KEYSI_OUTPUT_DIR", PROJECT_ROOT / "KeySI_results")).resolve())
RANDOM_SEED = int(os.getenv("KEYSI_RANDOM_SEED", "42"))

logging.basicConfig(
    level=getattr(logging, os.getenv("KEYSI_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("keysi")
logging.getLogger("werkzeug").setLevel(
    getattr(logging, os.getenv("KEYSI_REQUEST_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
)


def set_reproducibility_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        logger.debug("Could not set deterministic cuDNN flags.", exc_info=True)


set_reproducibility_seed()

training_in_progress = False
FINETUNE_LAST_LOSS = None
CURRENT_USER_NAME = "Yan"
_BM25_CACHE = {
    "csv_mtime": None,
    "bm25": None,
    "valid_indices": None
}
_KEYWORD_MATCH_CACHE = {
    "csv_mtime": None,
    "keyword_to_indices": {}
}
_DOC_INDEX_CACHE = {
    "csv_mtime": None,
    "valid_indices": None,
    "valid_idx_to_doc2d_idx": None
}
_USER_DATA_CACHE = {
    "mtime": None,
    "data": None
}

FILE_PATHS = {
    # data file
    "csv_path": str(DATA_DIR / "risk_factors.csv"), 
    "final_list_path": f"{OUTPUT_DIR}/final_list.json",
    
    # result
    "bm25_search_results": f"{OUTPUT_DIR}/bm25_search_results.json",
    "filtered_group_assignment": f"{OUTPUT_DIR}/filtered_group_assignment.json", 
    "group_assignment": f"{OUTPUT_DIR}/group_assignment.json",
    "triplet_trained_encoder": f"{OUTPUT_DIR}/triplet_trained_encoder.pth",
    "gap_based_filter_results": f"{OUTPUT_DIR}/gap_based_filter_results.json",
    "clustering_evaluation": f"{OUTPUT_DIR}/clustering_evaluation.json",
    "triplet_run_stats": f"{OUTPUT_DIR}/triplet_run_stats.json",
    "training_group_info": f"{OUTPUT_DIR}/training_group_info.json",
    "embeddings_trained": f"{OUTPUT_DIR}/embeddings_trained.npy",
    "triplet_training_comparison": f"{OUTPUT_DIR}/triplet_training_comparison.png",
    "triplet_tsne_comparison": f"{OUTPUT_DIR}/triplet_tsne_comparison.json",
    "user_finetuned_list": f"{OUTPUT_DIR}/user_finetuned_list.json",
    "bert_finetuned": f"{OUTPUT_DIR}/bert_finetuned.pth",
    "keysi_user_data": f"{OUTPUT_DIR}/keysi_user_data.json",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def reset_user_data_on_start():
    if os.getenv("KEYSI_RESET_USER_DATA_ON_START", "0").lower() not in {"1", "true", "yes"}:
        return
    try:
        user_data_path = get_user_data_path()
        if user_data_path and os.path.exists(user_data_path):
            base = {
                "training_sessions": [],
                "refinement_changes": [],
                "gap_filter_applied_once": False
            }
            write_json_atomic(user_data_path, base)
            _USER_DATA_CACHE["mtime"] = None
            _USER_DATA_CACHE["data"] = None
    except Exception as e:
        logger.warning("Failed to reset user data on start.", exc_info=True)

def get_user_data_path(user_name=None):
    name = (user_name or CURRENT_USER_NAME or "Yan").strip()
    if not name:
        name = "Yan"
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
    return f"{OUTPUT_DIR}/{safe_name}_keysi_user_data.json"

def get_safe_user_name(user_name=None):
    name = (user_name or CURRENT_USER_NAME or "Yan").strip()
    if not name:
        name = "Yan"
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name)

def get_user_model_dir(user_name=None):
    safe_name = get_safe_user_name(user_name)
    model_dir = os.path.join(OUTPUT_DIR, safe_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_user_data_to_user_dir(user_name=None):
    try:
        user_data_path = get_user_data_path(user_name)
        if not user_data_path or not os.path.exists(user_data_path):
            return
        model_dir = get_user_model_dir(user_name)
        safe_name = get_safe_user_name(user_name)
        target_path = os.path.join(model_dir, f"{safe_name}_keysi_user_data.json")
        shutil.copy2(user_data_path, target_path)
    except Exception as e:
        logger.warning("Failed to save user data into user directory.", exc_info=True)


def get_valid_doc2d_index_map(df_obj):
    try:
        csv_path = FILE_PATHS.get("csv_path")
        csv_mtime = os.path.getmtime(csv_path) if csv_path and os.path.exists(csv_path) else None
        if _DOC_INDEX_CACHE["valid_indices"] is not None and _DOC_INDEX_CACHE["csv_mtime"] == csv_mtime:
            return _DOC_INDEX_CACHE["valid_indices"], _DOC_INDEX_CACHE["valid_idx_to_doc2d_idx"]
        valid_mask = df_obj.iloc[:, 1].notna()
        valid_indices = df_obj.index[valid_mask].tolist()
        valid_idx_to_doc2d_idx = {valid_idx: i for i, valid_idx in enumerate(valid_indices)}
        _DOC_INDEX_CACHE["csv_mtime"] = csv_mtime
        _DOC_INDEX_CACHE["valid_indices"] = valid_indices
        _DOC_INDEX_CACHE["valid_idx_to_doc2d_idx"] = valid_idx_to_doc2d_idx
        return valid_indices, valid_idx_to_doc2d_idx
    except Exception:
        logger.warning("Failed to build valid document index map.", exc_info=True)
        return [], {}

def reset_user_data_for_new_training():
    try:
        user_data_path = get_user_data_path()
        base = {
            "training_sessions": [],
            "refinement_changes": [],
            "gap_filter_applied_once": False
        }
        write_json_atomic(user_data_path, base)
        _USER_DATA_CACHE["mtime"] = None
        _USER_DATA_CACHE["data"] = None
    except Exception as e:
        logger.warning("Failed to reset user data for new training.", exc_info=True)


TRAINING_CONFIG = {
    "freeze_layers": 6,
    
    "triplet_epochs": 10,        
    "triplet_batch_size": 16,
    "triplet_margin": 1.2,
    "triplet_lr": 1e-5,

    "proto_epochs": 5,       
    "proto_batch_size": 64,
    "proto_lr": 1e-5,
    

    "gap_alpha": 0.5,            
    "gap_min_samples": 20,
    "gap_percentile_fallback": 20,
    "gap_floor_threshold": 0.05,
    "gap_mix_ratio": 0.3,
    "gap_exclude_concentration_threshold": 0.25,  
    
    "encoding_batch_size": 64,
    "max_length": 256,
    "tsne_max_iter": 500,
    

    "min_pos_per_group": 2,
    "num_pos_per_anchor": 2,
    "num_neg_per_anchor": 3,
    "min_per_group_prototype": 5,
    "ema_alpha": 0.1,           
}

def get_config(key, default=None):

    return TRAINING_CONFIG.get(key, default)



LOCKED_BERT_NAME = "bert-base-uncased"
LOCKED_BERT_HIDDEN = 768
LOCKED_PROJ_DIM = 256

def _assert_locked_checkpoint(state_dict: dict):
    """
    Hard safety checks for loading encoder checkpoints.
    If any check fails, raise AssertionError immediately.
    """
    assert isinstance(state_dict, dict), f"checkpoint must be a state_dict (dict), got {type(state_dict)}"
    assert "proj.weight" in state_dict, "checkpoint missing required key: proj.weight"
    assert tuple(state_dict["proj.weight"].shape) == (LOCKED_PROJ_DIM, LOCKED_BERT_HIDDEN), (
        f"proj.weight.shape must be ({LOCKED_PROJ_DIM}, {LOCKED_BERT_HIDDEN}), "
        f"got {tuple(state_dict['proj.weight'].shape)}"
    )


class SentenceEncoder(nn.Module):

    
    def __init__(self, device='cpu'):
        super().__init__()

        self.bert = BertModel.from_pretrained(LOCKED_BERT_NAME, low_cpu_mem_usage=False)
        self.hidden = self.bert.config.hidden_size
        assert self.hidden == LOCKED_BERT_HIDDEN, f"Expected hidden_size={LOCKED_BERT_HIDDEN}, got {self.hidden}"
        self.proj = nn.Linear(self.hidden, LOCKED_PROJ_DIM)
        self.out_dim = LOCKED_PROJ_DIM
        self.ln = nn.LayerNorm(self.out_dim)

        if device != 'cpu':
            self.to(device)

    def encode_tokens(self, tokens):
        out = self.bert(**tokens).last_hidden_state 
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        pooled = self.proj(pooled)
        if pooled.dim()==1: pooled = pooled.unsqueeze(0)
        pooled = self.ln(pooled)
        return nn.functional.normalize(pooled, p=2, dim=-1)


def ensure_nltk_data():

    required_packages = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab', 
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng'
    }
    
    for package_name, package_path in required_packages.items():
        try:
            nltk.data.find(package_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception as e:
               
                if package_name == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
                        

    additional_packages = ['stopwords', 'wordnet', 'omw-1.4']
    for package in additional_packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass  


ensure_nltk_data()






if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


if device == "cuda":

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def clear_gpu_memory():

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


if not os.path.exists(FILE_PATHS["csv_path"]):
    raise FileNotFoundError(
        f"Input CSV not found: {FILE_PATHS['csv_path']}. "
        "Place the dataset there or set KEYSI_DATA_DIR."
    )

max_d = 30
word_count_threshold = 2


GROUP_COLORS = {
    "Group 1": "#FF6B6B",  # Red
    "Group 2": "#32CD32",  # Lime Green  
    "Group 3": "#FF8C00",  # Dark Orange
    "Group 4": "#8B4513",  # Saddle Brown
    "Group 5": "#FFD700",  # Gold
    "Group 6": "#8A2BE2",  # Blue Violet
    "Group 7": "#DC143C",  # Crimson
    "Group 8": "#228B22",  # Forest Green
    "Group 9": "#FF1493",  # Deep Pink
    "Group 10": "#800080", # Purple
    "Exclude": "#A9A9A9",    # Dark Gray for exclusion group
}

def get_group_color(group_name):

    return GROUP_COLORS.get(group_name, "#808080")  


PLOT_STYLES = {

    "background": {
        "color": "#1f77b4",  
        "size": 8,
        "opacity": 0.8,
        "line_width": 0.5,
        "line_color": "white"
    },
    
    "core": {
        "color": "#FFD700",  
        "size": 14,
        "opacity": 1.0,
        "symbol": "star",
        "line_width": 2,
        "line_color": "white"
    },
  
    "center": {
        "size": 20,
        "opacity": 1.0,
        "symbol": "diamond",
        "line_width": 3,
        "line_color": "white"
    },
  
    "layout": {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "xaxis": {
            "showgrid": False,  
            "zeroline": False,  
            "showline": True,   
            "linecolor": "black",
            "linewidth": 2,
            "mirror": True      
        },
        "yaxis": {
            "showgrid": False,  
            "zeroline": False,  
            "showline": True,   
            "linecolor": "black",
            "linewidth": 2,
            "mirror": True     
        }
    }
}




def ensure_directories(): 
    directories = [
        OUTPUT_DIR,
        "Keyword_Group",
        str(DATA_DIR)
    ]
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


ensure_directories()


try:
    embedding_model_kw = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
except Exception as e:
    raise


_GLOBAL_DOCUMENT_EMBEDDINGS = None
_GLOBAL_DOCUMENT_TSNE = None
_GLOBAL_DOCUMENT_EMBEDDINGS_READY = False

def precompute_document_embeddings():
   
    global _GLOBAL_DOCUMENT_EMBEDDINGS, _GLOBAL_DOCUMENT_TSNE, _GLOBAL_DOCUMENT_EMBEDDINGS_READY, df
    
    if _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        return
    
    
    try:
  
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        

        df_clean = df.dropna(subset=[df.columns[1]])
        all_articles_text = df_clean.iloc[:, 1].astype(str).tolist()
        valid_indices = df_clean.index.tolist()

        truncated_articles = [truncate_text_for_model(text, max_length=256) for text in all_articles_text]
        
        encoder = SentenceEncoder(device=device)
        encoder.eval()
        
        trained_model_path = FILE_PATHS["triplet_trained_encoder"]
        if os.path.exists(trained_model_path):
            try:
                state_dict = torch.load(trained_model_path, map_location=device)
                _assert_locked_checkpoint(state_dict)
                encoder.load_state_dict(state_dict, strict=True)
                encoder.eval()
            except Exception as e:
                logger.warning("Failed to load trained encoder checkpoint; using base encoder.", exc_info=True)
        else:
            logger.info("No trained encoder checkpoint found; using base encoder.")
        
        tokenizer = BertTokenizer.from_pretrained(LOCKED_BERT_NAME)
        
        batch_size = 64 if device == "cpu" else 128
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(truncated_articles), batch_size):
                batch_texts = truncated_articles[i:i + batch_size]
                
                tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                if device != 'cpu':
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                
                batch_embeddings = encoder.encode_tokens(tokens).cpu().numpy()
                all_embeddings.extend(batch_embeddings)
        
        _GLOBAL_DOCUMENT_EMBEDDINGS = np.array(all_embeddings)
        
        assert len(_GLOBAL_DOCUMENT_EMBEDDINGS) == len(df_clean), f"Embeddings length {len(_GLOBAL_DOCUMENT_EMBEDDINGS)} != clean df length {len(df_clean)}"
        
        
        if np.isnan(_GLOBAL_DOCUMENT_EMBEDDINGS).any():
            logger.warning("Document embeddings contain NaN values.")
        if np.isinf(_GLOBAL_DOCUMENT_EMBEDDINGS).any():
            logger.warning("Document embeddings contain infinite values.")
        
        n_samples = len(_GLOBAL_DOCUMENT_EMBEDDINGS)
        perplexity = min(30, max(5, n_samples // 3))
        perplexity = min(perplexity, n_samples - 1)
        
        import time
        start_time = time.time()
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=RANDOM_SEED,
            method='barnes_hut',
            angle=0.5,
            max_iter=300,
            verbose=0,
            n_jobs=-1
        )
        _GLOBAL_DOCUMENT_TSNE = tsne.fit_transform(_GLOBAL_DOCUMENT_EMBEDDINGS)
        elapsed = time.time() - start_time
        
        _GLOBAL_DOCUMENT_EMBEDDINGS_READY = True
        
        
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        _GLOBAL_DOCUMENT_EMBEDDINGS_READY = False
        logger.warning("Failed to precompute document embeddings.", exc_info=True)

def get_document_embeddings():
    
    global _GLOBAL_DOCUMENT_EMBEDDINGS, _GLOBAL_DOCUMENT_EMBEDDINGS_READY
    
    if not _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        precompute_document_embeddings()
    
    return _GLOBAL_DOCUMENT_EMBEDDINGS

def get_document_tsne():
    
    global _GLOBAL_DOCUMENT_TSNE, _GLOBAL_DOCUMENT_EMBEDDINGS_READY
    
    if not _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        precompute_document_embeddings()
    
    if _GLOBAL_DOCUMENT_TSNE is None:
        raise ValueError("t-SNE not pre-computed. This should not happen if precompute_document_embeddings() completed successfully.")
    
    return _GLOBAL_DOCUMENT_TSNE

def truncate_text_for_model(text, max_length=256):
 
    if not text or len(text) <= max_length:
        return text
    
  
    truncated = text[:max_length]
    

    if ' ' in truncated:
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  
            truncated = truncated[:last_space]
    
    return truncated + "..." if len(truncated) < len(text) else truncated

def contains_keyword_word_boundary(text, keyword):

    if not text or not keyword:
        return False

    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer('english')

    keyword_stem = stemmer.stem(keyword.lower())

    try:
        words = word_tokenize(text.lower())
        for word in words:
            word_stem = stemmer.stem(word)
            if word_stem == keyword_stem:
                return True
    except Exception as e:

        try:
            text_stemmed = stemmer.stem(text.lower())
            if keyword_stem in text_stemmed:
                return True
        except:
            pass
    
    return False



try:
    precompute_document_embeddings()
except Exception as e:
    logger.warning("Initial document embedding precompute failed.", exc_info=True)
        


kw_model = KeyBERT(model=embedding_model_kw)


word_count = Counter()
original_form = {}
df = pd.read_csv(FILE_PATHS["csv_path"])
all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
labels = df.iloc[:, 0].values


def preprocess_articles_batch(articles):

    processed_articles = []
    valid_indices = []
    
    for i, article in enumerate(articles):
        try:
            article = re.sub(r'\d+', '', article).strip()
            if len(article.split()) < 5:
                continue
            words = word_tokenize(article)
            tagged_words = pos_tag(words)
            nouns = [word for word, pos in tagged_words if pos.startswith("NN")]
            if nouns:
                processed_articles.append(" ".join(nouns))
                valid_indices.append(i)
        except Exception as e:
            continue
    
    return processed_articles, valid_indices

def extract_keywords_batch_gpu(articles, batch_size=None):

    if batch_size is None:
    
        if device == "cuda":
            batch_size = 128  
          
        else:
            batch_size = 32
    
    results = []
    total_batches = (len(articles) + batch_size - 1) // batch_size
    
  
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        
        try:
            with torch.no_grad():
                batch_embeddings = embedding_model_kw.encode(
                    batch, 
                    batch_size=len(batch),  
                    convert_to_tensor=True,
                    device=device
                )
            

            batch_results = []
            for i, article in enumerate(batch):
                keywords_info = kw_model.extract_keywords(
                    article, 
                    keyphrase_ngram_range=(1, 1), 
                    stop_words='english', 
                    top_n=8
                )
                stemmer = SnowballStemmer('english')
                result = [(stemmer.stem(kw), kw) for kw, _ in keywords_info]
                batch_results.append(result if result else None)
            
            results.extend(batch_results)
            
            clear_gpu_memory()
            
        except Exception as e:
            results.extend([None] * len(batch))
    
    return results

processed_articles, valid_indices = preprocess_articles_batch(all_articles_text)

if processed_articles:
    batch_results = extract_keywords_batch_gpu(processed_articles, batch_size=128)
    
    results = [None] * len(all_articles_text)
    for i, result in enumerate(batch_results):
        if i < len(valid_indices):
            results[valid_indices[i]] = result
else:
    results = [None] * len(all_articles_text)
for res in results:
    if res:
        for stemmed, kw in res:
            word_count[stemmed] += 1
            if stemmed not in original_form or len(kw) < len(original_form[stemmed]):
                original_form[stemmed] = kw

filtered_keywords = [original_form[stem] for stem, count in word_count.items() if count >= word_count_threshold]
if not filtered_keywords:
    raise ValueError("No keywords found with the specified frequency threshold.")

keyword_embeddings = embedding_model_kw.encode(filtered_keywords, convert_to_tensor=True).to(device).cpu().numpy()

n_keywords = len(keyword_embeddings)
perplexity = min(30, max(5, n_keywords // 3))
perplexity = min(perplexity, n_keywords - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED, method='barnes_hut', angle=0.5, max_iter=300, verbose=0)
reduced_embeddings = tsne.fit_transform(keyword_embeddings)

linkage_matrix = linkage(reduced_embeddings, method="ward")
labels_hierarchical = fcluster(linkage_matrix, max_d, criterion="distance")

clustered_keywords = {}

for word, label in zip(filtered_keywords, labels_hierarchical):
    clustered_keywords.setdefault(label, []).append(word)
best_k = len(set(labels_hierarchical))
output_dict = {f"cluster{cluster}": clustered_keywords[cluster] for cluster in sorted(clustered_keywords.keys())}
keywords = [kw for cluster in output_dict.values() for kw in cluster]
cluster_names = list(output_dict.keys())
total_clusters = len(cluster_names)
GLOBAL_OUTPUT_DICT = output_dict
GLOBAL_KEYWORDS = keywords  

if 'keywords' not in locals():
    keywords = []
