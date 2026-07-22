
import os
import shutil
import re 
import json
import json as json_module
import random
import math
import gc
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "KeySI_results")

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
    "csv_path": "CSV/risk_factors.csv", 
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
        pass

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
        pass


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
        pass


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

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def clear_gpu_memory():

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


if not os.getcwd().endswith("CSV") and not os.path.exists("CSV"):
    exit(1)

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
        "CSV"
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
                pass
        else:
            pass
        
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
            pass
        if np.isinf(_GLOBAL_DOCUMENT_EMBEDDINGS).any():
            pass
        
        n_samples = len(_GLOBAL_DOCUMENT_EMBEDDINGS)
        perplexity = min(30, max(5, n_samples // 3))
        perplexity = min(perplexity, n_samples - 1)
        
        import time
        start_time = time.time()
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
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
    pass
        


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
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, method='barnes_hut', angle=0.5, max_iter=300, verbose=0)
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

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .training-button {
                animation: pulse 1.5s infinite !important;
            }
            
            /* Article item click animation */
            .article-item-button {
                transition: all 0.2s ease !important;
            }
            
            .article-item-button:active {
                transform: scale(0.95) !important;
                background-color: #e3f2fd !important;
                border-color: #2196F3 !important;
                box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3) !important;
            }
            
            .article-item-button:hover {
                transform: scale(1.02) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            }
            
            /* Finetune documents list hover effect */
            .finetune-doc-card {
                transition: all 0.3s ease !important;
            }
            
            .finetune-doc-card:hover {
                transform: translateY(-4px) !important;
                box-shadow: 0 6px 16px rgba(0,0,0,0.12) !important;
            }
            
            .finetune-doc-card:active {
                transform: translateY(-2px) !important;
            }
            
            /* Button hover effects */
            button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            }
            
            /* Input focus effects */
            input:focus {
                outline: none !important;
                border-color: #3498db !important;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
            }
            
            /* Modern card styling */
            .modern-card {
                background: white !important;
                border-radius: 10px !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
                border: 1px solid #e9ecef !important;
                transition: all 0.3s ease !important;
            }
            
            .modern-card:hover {
                box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
                transform: translateY(-2px) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_layout():
    return html.Div([
        html.Div([
            html.H1("KeySI System", style={
                "textAlign": "center",
                "color": "#2c3e50",
                "fontSize": "2.5rem",
                "fontWeight": "bold",
                "marginBottom": "10px",
                "textShadow": "2px 2px 4px rgba(0,0,0,0.1)"
            }),
            html.Div([
                html.P(" Keyword System", style={
                    "textAlign": "center",
                    "color": "#7f8c8d",
                    "fontSize": "1.1rem",
                    "marginBottom": "0",
                    "fontStyle": "italic"
                }),
                html.Div([
                    html.Label("User Name:", style={
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                        "marginRight": "8px",
                        "fontSize": "0.95rem"
                    }),
                    dcc.Input(
                        id="user-name",
                        type="text",
                        value="Yan",
                        debounce=True,
                        style={
                            "padding": "6px 10px",
                            "borderRadius": "6px",
                            "border": "1px solid #ccc",
                            "minWidth": "160px"
                        }
                    )
                ], style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "justifyContent": "center",
                    "marginTop": "10px"
                })
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "gap": "8px",
                "marginBottom": "30px"
            })
        ], style={
            "backgroundColor": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "30px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.1)"
        }),
        
        
        html.Div([
        
            html.Div([
                html.Label("Number of Groups:", style={
                    "fontWeight": "bold",
                    "color": "#2c3e50",
                    "marginRight": "10px",
                    "fontSize": "1rem"
                }),
                dcc.Input(
                    id="group-count", 
                    type="number", 
                    value=3, 
                    min=1, 
                    step=1,
                    style={
                        "width": "80px",
                        "padding": "8px 12px",
                        "border": "2px solid #e0e0e0",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "marginRight": "15px"
                    }
                ),
                html.Button(
                    "Generate Groups", 
                    id="generate-btn", 
                    n_clicks=0,
                    style={
                        "backgroundColor": "#3498db",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "transition": "all 0.3s ease",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                        "marginRight": "10px",
                        "minWidth": "140px",
                        "flexShrink": "0"
                    }
                ),
                html.Button("Switch to Training View", id="switch-view-btn", n_clicks=0, style={
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "marginRight": "10px",
                    "minWidth": "180px",
                    "flexShrink": "0",
                    "display": "none"
                }),
                html.Button("Train Model", id="train-btn", n_clicks=0, style={
                    "backgroundColor": "#e74c3c",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "minWidth": "120px",
                    "flexShrink": "0"
                }),
                html.Button("Switch to Refinement Mode", id="switch-finetune-btn", n_clicks=0, style={
                    "backgroundColor": "#8e44ad",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "minWidth": "180px",
                    "flexShrink": "0",
                    "display": "none"
                })
            ], style={
                "display": "flex",
                "alignItems": "center",
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                "border": "1px solid #e9ecef",
                "width": "100%",
                "marginBottom": "20px",
                "flexWrap": "nowrap",
                "justifyContent": "flex-start"
            }),

            html.Div(id="gap-filter-warning", style={
                "color": "#c0392b",
                "fontWeight": "bold",
                "marginBottom": "15px",
                "textAlign": "center"
            }),
            
           
            html.Div([
                html.Label("Add Custom Keyword:", style={
                    "fontWeight": "bold",
                    "color": "#2c3e50",
                    "marginRight": "10px",
                    "fontSize": "1rem"
                }),
                dcc.Input(
                    id='new-keyword-input',
                    type='text',
                    placeholder='Enter keywords (use comma to separate multiple)...',
                    style={
                        "flex": "1",
                        "padding": "8px 12px",
                        "border": "2px solid #e0e0e0",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "marginRight": "15px"
                    }
                ),
                html.Button(
                    "Add Keyword", 
                    id="add-keyword-btn", 
                    n_clicks=0,
                    style={
                        "backgroundColor": "#27ae60",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "transition": "all 0.3s ease",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
                    }
                )
            ], style={
                "display": "flex",
                "alignItems": "center",
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                "border": "1px solid #e9ecef",
                "width": "48%",
                "marginLeft": "2%"
            })
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "marginBottom": "30px"
        }),

      
        dcc.Store(id="group-data", data={kw: None for kw in (keywords if 'keywords' in globals() else [])}),
        dcc.Store(id="selected-group", data=None),
        dcc.Store(id="group-order", data={}),
        dcc.Store(id="selected-file", data=None),
        dcc.Store(id="selected-keyword", data=None),
        dcc.Store(id="selected-article", data=None),  
        dcc.Store(id="articles-data", data=[]),  
        dcc.Store(id="document-embeddings", data=None), 
        dcc.Store(id="training-status", data={"is_training": False, "status": "idle"}),  
        dcc.Store(id="display-mode", data="keywords"),  
        dcc.Store(id="training-figures", data={"before": None, "after": None}),  
        dcc.Store(id="user-name-store", data="Yan"),
        dcc.Store(id="highlighted-indices", data=[]),  
        dcc.Store(id="keyword-highlights", data=[]),  
        dcc.Store(id="training-selected-group", data=None),  
        dcc.Store(id="training-selected-keyword", data=None),  
        dcc.Store(id="training-selected-article", data=None),  
        # Finetune mode stores
        dcc.Store(id="finetune-figures", data=None),  
        dcc.Store(id="finetune-selected-group", data=None),
        dcc.Store(id="finetune-selected-sample", data=None),
        dcc.Store(id="finetune-selected-keyword", data=None),  
        dcc.Store(id="finetune-selected-article-index", data=None),  
        dcc.Store(id="finetune-highlight-core", data=[]),
        dcc.Store(id="finetune-temp-assignments", data={}),
        
        html.Div(id="main-visualization-area", children=[
           
            html.Div([
                html.H4("Keywords 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Click on keywords to view related documents", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='keywords-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '1%'
            }),
            
           
            html.Div([
                html.H4("Documents 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Documents highlighted by selected keyword", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='documents-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '1%'
            })
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        
        html.Div([
            html.Div([
                html.H4("Group Management", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="group-containers", style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "15px",
                    "margin-bottom": "20px"
                }),
            ], className="modern-card", style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '15px'
            }),
            
        
            html.Div([
                html.H4("Recommended Articles", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="articles-container", style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'margin': '0 7px'
            }),
            
            
            html.Div([
                html.H4("Article Full Text", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="article-fulltext-container", children=[
                    html.P("Click on an article from the middle panel to view its full content", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '30%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '7px'
            })
        ], id="keywords-group-management-area", style={'display': 'flex', 'marginBottom': '30px', 'height': '600px', 'overflowY': 'auto'}),
        
        
        html.Div(id="training-group-management-area", style={'display': 'none', 'marginBottom': '30px', 'height': '600px', 'overflowY': 'auto'}, children=[
            html.Div([
                html.H4("Training Group Management", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="training-group-containers", children=[
                    html.P("Loading training groups...", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "15px",
                    "margin-bottom": "20px"
                }),
            ], className="modern-card", style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H4("Training Recommended Articles", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="training-articles-container", children=[
                    html.P("Loading training articles...", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'margin': '0 7px'
            }),
            
            html.Div([
                html.H4("Training Article Full Text", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="training-article-fulltext-container", children=[
                    html.P("Click on an article from the middle panel to view its full content", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '30%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '7px'
            })
        ]),
        
        html.Div(
            id="finetune-group-management-area",
            style={'display': 'none', 'marginBottom': '30px', 'height': '600px', 'overflowY': 'auto'},
            children=[
                html.Div([
                    html.H4("Refinement Group Management", style={
                        "color": "#2c3e50",
                        "fontSize": "1.3rem",
                        "fontWeight": "bold",
                        "marginBottom": "15px",
                        "textAlign": "center"
                    }),
                    html.Div(id="finetune-group-containers", children=[
                        html.P("Loading refinement groups...",
                               style={
                                   "color": "#7f8c8d",
                                   "fontStyle": "italic",
                                   "textAlign": "center",
                                   "padding": "40px 20px",
                                   "fontSize": "1rem"
                               })
                    ], style={
                        "display": "flex",
                        "flex-direction": "column",
                        "gap": "15px",
                        "margin-bottom": "20px"
                    }),
                ], className="modern-card", style={
                    'width': '25%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '20px',
                    'marginRight': '15px'
                }),
                html.Div([
                    html.H4("Documents List", id="finetune-articles-title", style={
                        "color": "#2c3e50",
                        "fontSize": "1.3rem",
                        "fontWeight": "bold",
                        "marginBottom": "15px",
                        "textAlign": "center"
                    }),
                    html.Div(id="finetune-articles-container", children=[
                        html.P("Select a group or keyword to view documents",
                               style={
                                   "color": "#7f8c8d",
                                   "fontStyle": "italic",
                                   "textAlign": "center",
                                   "padding": "40px 20px",
                                   "fontSize": "1rem"
                               })
                    ], style={
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "8px",
                        "padding": "20px",
                        "minHeight": "500px",
                        "maxHeight": "700px",
                        "overflowY": "auto",
                        "border": "1px solid #e9ecef"
                    })
                ], className="modern-card", style={
                    'width': '40%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '20px',
                    'margin': '0 7px'
                }),
                html.Div([
                    html.H4("Sample Operations", style={
                        "color": "#2c3e50",
                        "fontSize": "1.2rem",
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                        "textAlign": "center"
                    }),
                    html.Div(id="finetune-operation-buttons", children=[], style={"marginBottom": "20px"}),
                    html.H5("Article Full Text", style={
                        "color": "#2c3e50",
                        "fontSize": "1.1rem",
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                        "textAlign": "center"
                    }),
                    html.Div(id="finetune-text-container", children=[
                        html.P("Click a document to preview",
                               style={
                                   "color": "#7f8c8d",
                                   "fontStyle": "italic",
                                   "textAlign": "center",
                                   "padding": "20px",
                                   "fontSize": "0.9rem"
                               })
                    ], style={
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "8px",
                        "padding": "15px",
                        "minHeight": "150px",
                        "maxHeight": "200px",
                        "overflowY": "auto",
                        "border": "1px solid #e9ecef",
                        "marginBottom": "20px",
                        "fontSize": "0.85rem"
                    }),
                    html.H4("Adjustment History", style={
                        "color": "#2c3e50",
                        "fontSize": "1.2rem",
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                        "textAlign": "center"
                    }),
                    html.Div(id="finetune-adjustment-history", children=[
                        html.P("No adjustments yet",
                               style={
                                   "color": "#7f8c8d",
                                   "fontStyle": "italic",
                                   "textAlign": "center",
                                   "padding": "15px",
                                   "fontSize": "0.85rem"
                               })
                    ], style={
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "8px",
                        "padding": "15px",
                        "minHeight": "180px",
                        "maxHeight": "250px",
                        "overflowY": "auto",
                        "border": "1px solid #e9ecef",
                        "marginBottom": "15px",
                        "fontSize": "0.85rem"
                    }),
                    html.Div(id="finetune-history-buttons", children=[], style={"marginTop": "10px"}),
                    html.Div([
                        html.Button("Run Refinement Training", id="finetune-train-btn", n_clicks=0, style={
                            "backgroundColor": "#9b59b6",
                            "color": "white",
                            "border": "none",
                            "padding": "12px 24px",
                            "borderRadius": "6px",
                            "fontSize": "1rem",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "transition": "all 0.3s ease",
                            "boxShadow": "0 3px 8px rgba(155, 89, 182, 0.3)",
                            "width": "100%",
                            "marginBottom": "10px"
                        }),
                        html.Div(id="finetune-training-status", style={
                            "marginTop": "10px",
                            "textAlign": "center",
                            "fontWeight": "bold"
                        }),
                        html.Button("Clear Adjustment History", id="finetune-clear-history-btn", n_clicks=0, style={
                            "backgroundColor": "#e74c3c",
                            "color": "white",
                            "border": "none",
                            "padding": "8px 16px",
                            "borderRadius": "6px",
                            "fontSize": "0.9rem",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "transition": "all 0.3s ease",
                            "boxShadow": "0 2px 5px rgba(231, 76, 60, 0.3)",
                            "width": "100%"
                        })
                    ])
                ], className="modern-card", style={
                    'width': '30%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '20px',
                    'marginLeft': '7px'
                })
            ]
        ),
        

        html.Div(id="status-output", style={"marginTop": "20px"}),
        

        dcc.Interval(
            id="interval-component",
            interval=1000, 
            n_intervals=0
        )
    ])

app.layout = create_layout()

@app.callback(
    Output("group-order", "data"),
    [
        Input("generate-btn", "n_clicks"),
        Input("group-data", "data"),
    ],
    [
        State("group-count", "value"),
        State("group-order", "data")
    ],
    prevent_initial_call=True
)
def update_group_order(generate_n_clicks, group_data, num_groups, current_order):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    new_order = dict(current_order) if current_order else {}
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "generate-btn":
        if not num_groups or num_groups < 1:
            raise PreventUpdate
        
        groups = {f"Group {i+1}": [] for i in range(num_groups)}
        
        groups["Exclude"] = []
        update_live_keywords_snapshot(groups)
        return groups
    
    elif triggered_id == "group-data":
        for group_name in new_order:
            new_order[group_name] = []
        
        for kw, grp in group_data.items():
            if grp and grp in new_order:
                new_order[grp].append(kw)
        update_live_keywords_snapshot(new_order)
        return new_order

    return new_order

@app.callback(
    Output("group-containers", "children"),
    [Input("group-order", "data"),
     Input("selected-group", "data"),
     Input("selected-keyword", "data")], 
    [State("display-mode", "data")]
)
def render_groups(group_order, selected_group, selected_keyword, display_mode):
    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
        if grp_name == "Exclude":
            group_display_name = "Exclude"
            group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Group {group_number}"
            group_color = get_group_color(grp_name)
        
        if grp_name == "Exclude":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  
            }
        else:
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px solid {group_color}",
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px"
            }
        
        group_header = html.Button(
            group_display_name,
            id={"type": "group-header", "index": grp_name},
            style=header_style
        )

        group_keywords = []
        for i, kw in enumerate(kw_list):
            
            if display_mode == "training":
                is_selected = False  
                is_group_selected = False
            else:
                is_selected = selected_keyword and kw == selected_keyword   
                is_group_selected = selected_group and grp_name == selected_group
            
            if grp_name == "Exclude" and is_group_selected:
                keyword_bg_color = "#808080"  
                keyword_text_color = "white"
                keyword_border = "1px solid #808080"
                keyword_font_weight = "bold"
            elif is_selected:
                keyword_bg_color = group_color
                keyword_text_color = "white"
                keyword_border = f"1px solid {group_color}"
                keyword_font_weight = "bold"
            else:
                keyword_bg_color = f"{group_color}20"
                keyword_text_color = group_color
                keyword_border = f"1px solid {group_color}"
                keyword_font_weight = "normal"
            
            keyword_button = html.Button(
                kw,
                id={"type": "select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": keyword_border, 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": keyword_bg_color,  
                    "color": keyword_text_color,  
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": keyword_font_weight  
                }
            )
            
            keyword_item = html.Div([
                keyword_button,
                html.Button("×", id={"type": "remove-keyword", "group": grp_name, "index": i}, 
                           style={"margin": "2px", "padding": "2px 6px", "fontSize": "10px", "color": "red", "float": "right"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"})
            group_keywords.append(keyword_item)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "8px",
                "minHeight": "50px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "10px"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"marginBottom": "15px"}
        )
        children.append(group_container)

    return children


@app.callback(
    [Output("selected-group", "data"),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input({"type": "group-header", "index": ALL}, "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def select_group(n_clicks, display_mode):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    triggered_prop_id = ctx.triggered[0].get('prop_id', 'N/A')
    triggered_value = ctx.triggered[0].get('value', 'N/A')
    
    if "group-header" in triggered_id and triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
        try:
            parsed_id = json.loads(triggered_id.split('.')[0])
            selected_group = parsed_id["index"]
            
            if display_mode == "training":
                return selected_group, dash.no_update
            else:
                return selected_group, None
                
        except Exception as e:
            traceback.print_exc()
            raise PreventUpdate

    raise PreventUpdate

@app.callback(
    [Output("selected-keyword", "data", allow_duplicate=True),
     Output("keyword-highlights", "data", allow_duplicate=True),
     Output("selected-group", "data", allow_duplicate=True)],
    [Input({"type": "select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def select_keyword_from_group(n_clicks, display_mode, group_order):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    if "select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            if triggered_n_clicks is None:
                raise PreventUpdate
            
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                if display_mode == "training":
                    keyword_docs = []
                    
                    try:
                        global df
                        if 'df' not in globals():
                            df = pd.read_csv(FILE_PATHS["csv_path"])
                        
                        for i in range(len(df)):
                            text = str(df.iloc[i, 1])
                            if contains_keyword_word_boundary(text, keyword):
                                keyword_docs.append(i)
                        
                        return dash.no_update, keyword_docs, dash.no_update
                    except Exception as e:
                        return dash.no_update, [], dash.no_update
                else:
                    return keyword, dash.no_update, None
            else:
                raise PreventUpdate
            
        except Exception as e:
            raise PreventUpdate
    raise PreventUpdate

@app.callback(
    [Output("group-order", "data", allow_duplicate=True),
     Output("group-data", "data", allow_duplicate=True)],
    Input({"type": "remove-keyword", "group": ALL, "index": ALL}, "n_clicks"),
    [State("group-order", "data"),
     State("group-data", "data")],
    prevent_initial_call=True
)
def remove_keyword_from_group(n_clicks, group_order, group_data):
    ctx = dash.callback_context
    
    if not ctx.triggered or not any(n_clicks):
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    if not triggered_id or '.n_clicks' not in triggered_id:
        raise PreventUpdate
    
    try:
        button_id = json.loads(triggered_id.split('.')[0])
        group_name = button_id.get("group")
        keyword_index = button_id.get("index")
        
        if not group_name or keyword_index is None:
            raise PreventUpdate
        
        new_group_order = dict(group_order) if group_order else {}
        
        if group_name in new_group_order:
            keyword_list = list(new_group_order[group_name])
            if 0 <= keyword_index < len(keyword_list):
                removed_keyword = keyword_list.pop(keyword_index)
                new_group_order[group_name] = keyword_list
                
                new_group_data = dict(group_data) if group_data else {}
                if removed_keyword in new_group_data:
                    new_group_data[removed_keyword] = None
                clear_caches()
                update_live_keywords_snapshot(new_group_order)
                return new_group_order, new_group_data
        else:
            pass
            
    except Exception:
        pass
    raise PreventUpdate

@app.callback(
    Output("articles-container", "children"),
    [Input("selected-keyword", "data"),
     Input("selected-group", "data")],  
    [State("group-order", "data"),  
     State("display-mode", "data")],
    prevent_initial_call=True
)
def display_recommended_articles(selected_keyword, selected_group, group_order, display_mode):
    try:
        global df, _ARTICLES_CACHE
        if 'df' not in globals():
            return html.P("Data not loaded")
        cache_key = None
        if selected_keyword:
            cache_key = f"keyword:{selected_keyword}"
        elif selected_group and group_order:
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    cache_key = f"group:{group_name}:{':'.join(sorted(keywords))}"
                    break
        user_data_mtime = get_keysi_user_data_mtime()
        if cache_key and user_data_mtime:
            cache_key = f"{cache_key}:ud:{user_data_mtime}"
        
        if cache_key and cache_key in _ARTICLES_CACHE:
            return _ARTICLES_CACHE[cache_key]
        
        search_keywords = []
        search_title = ""
        use_snapshot = False
        use_preselected_indices = False
        preselected_indices = []
        snapshot_before = get_latest_training_snapshot("before")
        deduped_group_docs = None
        deduped_keyword_matches_in_group = None
        if snapshot_before and snapshot_before.get("group_docs"):
            deduped_group_docs = dedupe_group_docs_by_priority(snapshot_before.get("group_docs", {}), group_order)
            if snapshot_before.get("keyword_matches_in_group"):
                deduped_keyword_matches_in_group = filter_keyword_matches_in_group(
                    snapshot_before.get("keyword_matches_in_group", {}), deduped_group_docs
                )
        
        if selected_keyword:
            search_keywords = [selected_keyword]
            search_title = f"Articles containing '{selected_keyword}'"
            if selected_group:
                pass
            if deduped_keyword_matches_in_group is not None:
                keyword_group = resolve_group_for_keyword(group_order, selected_keyword)
                if keyword_group:
                    preselected_indices = deduped_keyword_matches_in_group.get(keyword_group, {}).get(selected_keyword, [])
                    use_snapshot = True
                    search_title = f"Articles containing '{selected_keyword}' (training before snapshot)"
        elif selected_group:
            if deduped_group_docs is not None:
                preselected_indices = deduped_group_docs.get(selected_group, [])
                use_snapshot = True
                search_title = f"Articles in {selected_group} (training before snapshot)"
            if selected_group == "Exclude" and (not preselected_indices) and snapshot_before and snapshot_before.get("keyword_matches_all_docs"):
                exclude_matches = snapshot_before["keyword_matches_all_docs"].get("Exclude", {})
                if exclude_matches:
                    merged = set()
                    for _, idxs in exclude_matches.items():
                        merged.update(idxs)
                    preselected_indices = sorted(merged)
                    use_snapshot = True
                    search_title = "Articles in Exclude group (training before snapshot)"

            if selected_group == "Exclude" and not use_snapshot:
                search_title = "Articles in Exclude group"
                try:
                    filtered_path = FILE_PATHS.get("filtered_group_assignment")
                    if filtered_path and os.path.exists(filtered_path):
                        with open(filtered_path, "r", encoding="utf-8") as f:
                            filtered_dict = json.load(f)
                        if "Exclude" in filtered_dict:
                            preselected_indices = filtered_dict.get("Exclude", [])
                            use_preselected_indices = True
                        else:
                            return html.Div([
                                html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                                html.P("Exclude group not found in training results", 
                                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                            ])
                    else:
                        return html.Div([
                            html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                            html.P("No training results available. Please run training first.", 
                                   style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                        ])
                except Exception as e:
                    return html.Div([
                        html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.P(f"Error loading Exclude group: {str(e)}", 
                               style={"color": "#e74c3c", "textAlign": "center", "padding": "20px"})
                    ])
            
            if group_order and not use_preselected_indices:
                search_keywords = []
                for group_name, keywords in group_order.items():
                    if group_name == selected_group:
                        search_keywords = keywords
                        break
                
                if search_keywords:
                    search_title = f"Articles containing keywords from group '{selected_group}'"
                else:
                    return html.Div([
                        html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.P(f"Group '{selected_group}' has no keywords assigned", 
                               style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                    ])
            elif not use_preselected_indices:
                return html.Div([
                    html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                    html.P("No groups have been created yet", 
                           style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                ])
        else:
            return html.Div([
                html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("Please select a keyword or group to view recommended articles", 
                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
            ])
        
        matching_articles = []

        if use_snapshot:
            use_preselected_indices = True
        if use_preselected_indices:
            for idx in preselected_indices:
                if idx < len(df):
                    text = str(df.iloc[idx, 1])
                    file_keywords = extract_top_keywords(text, 5)
                    matching_articles.append({
                        'file_number': idx + 1,
                        'file_index': idx,
                        'text': text,
                        'keywords': file_keywords,
                        'bm25_score': 1.0
                    })
            if use_snapshot:
                pass
            else:
                pass
        else:
            if selected_keyword:
                cached_indices = get_keyword_doc_indices_cached(selected_keyword, df)
                for idx in cached_indices:
                    if idx < len(df):
                        text = str(df.iloc[idx, 1])
                        file_keywords = extract_top_keywords(text, 5)
                        matching_articles.append({
                            'file_number': idx + 1,
                            'file_index': idx,
                            'text': text,
                            'keywords': file_keywords,
                            'bm25_score': 1.0
                        })
            elif selected_group and group_order:
                group_keywords = group_order.get(selected_group, [])
                cached_indices = get_group_doc_indices_cached(group_keywords, df)
                for idx in cached_indices:
                    if idx < len(df):
                        text = str(df.iloc[idx, 1])
                        file_keywords = extract_top_keywords(text, 5)
                        matching_articles.append({
                            'file_number': idx + 1,
                            'file_index': idx,
                            'text': text,
                            'keywords': file_keywords,
                            'bm25_score': 1.0
                        })
        
        if not matching_articles:
            result = html.P(f"No articles found for the selected search criteria")
            if cache_key:
                _ARTICLES_CACHE[cache_key] = result
            return result
        
        article_items = [
            html.H6(f"{search_title} (Found {len(matching_articles)} articles)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        
        for article_info in matching_articles:
            keyword_tags = []
            for keyword in article_info['keywords']:
                keyword_tag = html.Span(
                    keyword,
                    style={
                        "backgroundColor": "#e3f2fd",
                        "color": "#1976d2",
                        "padding": "2px 6px",
                        "margin": "2px",
                        "borderRadius": "12px",
                        "fontSize": "11px",
                        "display": "inline-block"
                    }
                )
                keyword_tags.append(keyword_tag)
            
            article_item = html.Div([
                html.Button(
                    html.Div([
                        html.H6(f"Article {article_info['file_number']}", 
                               style={"color": "#333", "marginBottom": "8px", "fontSize": "14px", "margin": "0"}),
                        html.Div([
                            html.Span("Top 5 Keywords: ", style={"fontWeight": "bold", "color": "#666"}),
                            html.Div(keyword_tags, style={"display": "inline-block", "marginLeft": "5px"})
                        ], style={"marginBottom": "8px"}),
                    ]),
                    id={"type": "article-item", "index": article_info['file_index']},
                    className="article-item-button",
                    style={
                        "width": "100%",
                        "padding": "12px", 
                        "border": "1px solid #eee",
                        "backgroundColor": "white",
                        "borderRadius": "5px",
                        "marginBottom": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                        "cursor": "pointer",
                        "textAlign": "left",
                        "outline": "none"
                    },
                    n_clicks=0
                ),
                html.Hr(style={"margin": "4px 0", "borderColor": "#ddd"})
            ])
            article_items.append(article_item)
        
        result = html.Div(article_items)
        if cache_key:
            _ARTICLES_CACHE[cache_key] = result
        
        return result
        
    except Exception as e:
        return html.P(f"Error displaying recommended articles: {str(e)}")

def extract_top_keywords(text, top_k=5):
    try:
        global kw_model
        if 'kw_model' in globals() and kw_model:
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), 
                                               stop_words='english')
            return [kw[0] for kw in keywords[:top_k]]
        else:
            words = text.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
            return filtered_words[:top_k]
    except Exception as e:
        return ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]

    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']

    if not triggered_n_clicks or triggered_n_clicks is None:
        raise PreventUpdate

    selected_group = json.loads(triggered_id.split('.')[0])["index"]
    
    return selected_group, None  

def record_user_data(action_type, group_order=None, matched_dict=None, refinement_change=None):

    try:
        import datetime
        
        def build_keyword_doc_matches(group_keywords, df_obj, restrict_groups=None):

            keyword_doc_matches = {}
            for grp_name, keywords in (group_keywords or {}).items():
                keyword_doc_matches[grp_name] = {}
                doc_pool = None
                if restrict_groups and grp_name in restrict_groups:
                    doc_pool = restrict_groups.get(grp_name, [])
                for kw in keywords:
                    matched = []
                    if doc_pool is None:
                        for i in range(len(df_obj)):
                            text = str(df_obj.iloc[i, 1])
                            if contains_keyword_word_boundary(text, kw):
                                matched.append(i)
                    else:
                        for i in doc_pool:
                            if i >= len(df_obj):
                                continue
                            text = str(df_obj.iloc[i, 1])
                            if contains_keyword_word_boundary(text, kw):
                                matched.append(i)
                    keyword_doc_matches[grp_name][kw] = matched
            return keyword_doc_matches
        
        user_data_path = get_user_data_path()
        if os.path.exists(user_data_path):
            try:
                with open(user_data_path, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
            except:
                user_data = {
                    "training_sessions": [],
                    "refinement_changes": []
                }
        else:
            user_data = {
                "training_sessions": [],
                "refinement_changes": []
            }
        
        if action_type == "training_before":
            global df
            if 'df' not in globals():
                df = pd.read_csv(FILE_PATHS["csv_path"])
            
            try:
                user_data_state = load_keysi_user_data()
                has_history = bool(user_data_state.get("training_sessions") or user_data_state.get("refinement_changes"))
                snapshot_for_training = get_latest_snapshot_for_training()
                if has_history and snapshot_for_training and snapshot_for_training.get("group_docs"):
                    matched_dict_before = {
                        g: list(idxs) for g, idxs in snapshot_for_training.get("group_docs", {}).items()
                    }
                    group_counts_before = {g: len(indices) for g, indices in matched_dict_before.items()}
                    keyword_matches_all = snapshot_for_training.get("keyword_matches_all_docs", {})
                    keyword_matches_in_group = snapshot_for_training.get("keyword_matches_in_group")
                    if keyword_matches_in_group is None:
                        keyword_matches_in_group = build_keyword_doc_matches(group_order, df, matched_dict_before)
                    keyword_counts = {}
                    for group_name, keywords in (group_order or {}).items():
                        for keyword in keywords:
                            keyword_counts[keyword] = len(keyword_matches_all.get(group_name, {}).get(keyword, []))
                    for g in (group_order or {}).keys():
                        if g not in matched_dict_before:
                            matched_dict_before[g] = []
                            group_counts_before[g] = 0
                else:
                    from nltk.stem import SnowballStemmer
                    from nltk.tokenize import word_tokenize
                    import re
                    
                    stemmer = SnowballStemmer('english')
                    
                    def process_articles_serial(articles):
                        tokenized_corpus, valid_indices = [], []
                        for i, a in enumerate(articles):
                            try:
                                a = re.sub(r'\d+', '', str(a)).strip()
                                if len(a.split()) >= 5:
                                    words = word_tokenize(a)
                                    stemmed = [stemmer.stem(w.lower()) for w in words]
                                    if stemmed:
                                        tokenized_corpus.append(" ".join(stemmed))
                                        valid_indices.append(i)
                            except Exception:
                                pass
                        return tokenized_corpus, valid_indices
                    
                    df_clean = df.dropna(subset=[df.columns[1]])
                    all_texts = df_clean.iloc[:, 1].astype(str).tolist()
                    bm25, valid_indices = get_bm25_cache(df)
                    if bm25 is None:
                        tokenized_corpus, valid_indices = process_articles_serial(all_texts)
                        bm25 = BM25Okapi([s.split() for s in tokenized_corpus])
                    
                    def bm25_search_batch(bm25, query_groups, valid_indices):
                        results = {}
                        for g, words in query_groups.items():
                            q = [stemmer.stem(w.lower()) for w in words]
                            scores = bm25.get_scores(q)
                            idx_corpus = [i for i, s in enumerate(scores) if s > 0.1]
                            if len(idx_corpus) == 0:
                                idx_corpus = [i for i, s in enumerate(scores) if s > 0.01]
                            idx_orig = [valid_indices[i] for i in idx_corpus]
                            results[g] = idx_orig[:3000]
                        return results
                    
                    bm25_groups = {g: kws for g, kws in (group_order or {}).items() if kws}
                    matched_dict_before = bm25_search_batch(bm25, bm25_groups, valid_indices)
                    for g in (group_order or {}).keys():
                        if g not in matched_dict_before:
                            matched_dict_before[g] = []
                    
                    keyword_counts = {}
                    for group_name, keywords in group_order.items():
                        for keyword in keywords:
                            count = 0
                            for i in range(len(df)):
                                text = str(df.iloc[i, 1])
                                if contains_keyword_word_boundary(text, keyword):
                                    count += 1
                            keyword_counts[keyword] = count
                    
                    group_counts_before = {g: len(indices) for g, indices in matched_dict_before.items()}
                    
                    keyword_matches_all = build_keyword_doc_matches(group_order, df)
                    keyword_matches_in_group = build_keyword_doc_matches(group_order, df, matched_dict_before)
                
            except Exception as e:
                keyword_counts = {}
                for group_name, keywords in group_order.items():
                    for keyword in keywords:
                        count = 0
                        for i in range(len(df)):
                            text = str(df.iloc[i, 1])
                            if contains_keyword_word_boundary(text, keyword):
                                count += 1
                        keyword_counts[keyword] = count
                
                group_counts_before = {}
                matched_dict_before = {}
                for group_name, keywords in group_order.items():
                    group_docs = set()
                    for keyword in keywords:
                        for i in range(len(df)):
                            text = str(df.iloc[i, 1])
                            if contains_keyword_word_boundary(text, keyword):
                                group_docs.add(i)
                    group_counts_before[group_name] = len(group_docs)
                    matched_dict_before[group_name] = list(group_docs)
                
                keyword_matches_all = build_keyword_doc_matches(group_order, df)
                keyword_matches_in_group = build_keyword_doc_matches(group_order, df, matched_dict_before)
            
            session_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "training",
                "before": {
                    "keywords": group_order,
                    "keyword_counts": keyword_counts,
                    "group_counts": group_counts_before,
                    "keyword_matches_all_docs": keyword_matches_all,
                    "keyword_matches_in_group": keyword_matches_in_group,
                    "group_docs": matched_dict_before
                }
            }
            user_data["training_sessions"] = [session_data]
            user_data["refinement_changes"] = []
            
        elif action_type == "training_after":
            if 'df' not in globals():
                df = pd.read_csv(FILE_PATHS["csv_path"])
            
            if user_data["training_sessions"]:
                latest_session = user_data["training_sessions"][-1]
                if latest_session.get("type") == "training" and "after" not in latest_session:
                    keyword_counts_after = {}
                    for group_name, keywords in group_order.items():
                        for keyword in keywords:
                            count = 0
                            for i in range(len(df)):
                                text = str(df.iloc[i, 1])
                                if contains_keyword_word_boundary(text, keyword):
                                    count += 1
                            keyword_counts_after[keyword] = count
                    
                    group_counts_after = {}
                    for g, indices in matched_dict.items():
                        if g == "Exclude":
                            if "before" in latest_session and "group_counts" in latest_session["before"]:
                                group_counts_after[g] = latest_session["before"]["group_counts"].get(g, len(indices))
                            else:
                                group_counts_after[g] = len(indices)
                        else:
                            group_counts_after[g] = len(indices)
                    
                    keyword_matches_all = build_keyword_doc_matches(group_order, df)
                    keyword_matches_in_group = build_keyword_doc_matches(group_order, df, matched_dict)
                    
                    latest_session["after"] = {
                        "keyword_counts": keyword_counts_after,
                        "group_counts": group_counts_after,
                        "matched_dict_sizes": {g: len(indices) for g, indices in matched_dict.items()},
                        "keyword_matches_all_docs": keyword_matches_all,
                        "keyword_matches_in_group": keyword_matches_in_group,
                        "group_docs": matched_dict
                    }
            
        elif action_type == "refinement_change":
            change_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "change": refinement_change
            }
            
            if group_order and matched_dict and 'df' in globals():
                try:
                    keyword_matches_all = build_keyword_doc_matches(group_order, df)
                    keyword_matches_in_group = build_keyword_doc_matches(group_order, df, matched_dict)
                    change_entry["after"] = {
                        "keywords": group_order,
                        "group_counts": {g: len(indices) for g, indices in matched_dict.items()},
                        "keyword_matches_all_docs": keyword_matches_all,
                        "keyword_matches_in_group": keyword_matches_in_group,
                        "group_docs": matched_dict
                    }
                except Exception as e:
                    pass
            
            if "refinement_changes" not in user_data or not isinstance(user_data["refinement_changes"], list):
                user_data["refinement_changes"] = []
            user_data["refinement_changes"].append(change_entry)
        
        write_json_atomic(user_data_path, user_data)
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()

def load_keysi_user_data():
    try:
        user_data_path = get_user_data_path()
        if user_data_path and os.path.exists(user_data_path):
            mtime = os.path.getmtime(user_data_path)
            if _USER_DATA_CACHE["data"] is not None and _USER_DATA_CACHE["mtime"] == mtime:
                return _USER_DATA_CACHE["data"]
            if os.path.getsize(user_data_path) == 0:
                return {}
            with open(user_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _USER_DATA_CACHE["mtime"] = mtime
                _USER_DATA_CACHE["data"] = data
                return data
    except Exception as e:
        pass
    return {}

def write_json_atomic(path, data):
    try:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        return False

def write_state_dict_atomic(path, state_dict):
    try:
        base_dir = os.path.dirname(path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        tmp_path = f"{path}.tmp"
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        return False

reset_user_data_on_start()

def get_keysi_user_data_mtime():
    try:
        user_data_path = get_user_data_path()
        if user_data_path and os.path.exists(user_data_path):
            return int(os.path.getmtime(user_data_path))
    except Exception:
        pass
    return None

def get_latest_training_snapshot(stage):
    user_data = load_keysi_user_data()
    sessions = user_data.get("training_sessions", [])
    if not sessions:
        return None
    latest = sessions[-1]
    return latest.get(stage)

def get_latest_refinement_snapshot():
    user_data = load_keysi_user_data()
    changes = user_data.get("refinement_changes", [])
    for entry in reversed(changes):
        if isinstance(entry, dict) and "after" in entry:
            return entry.get("after")
    return None

def resolve_group_for_keyword(group_order, keyword):
    for grp_name, keywords in (group_order or {}).items():
        if keyword in keywords:
            return grp_name
    return None

def dedupe_group_docs_by_priority(group_docs, group_order):
    if not group_docs:
        return {}
    ordered_groups = list((group_order or {}).keys()) or list(group_docs.keys())
    if "Exclude" in ordered_groups:
        ordered_groups = ["Exclude"] + [g for g in ordered_groups if g != "Exclude"]
    seen = set()
    deduped = {}
    for g in ordered_groups:
        indices = group_docs.get(g, [])
        deduped[g] = []
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            deduped[g].append(idx)
    for g, indices in group_docs.items():
        if g in deduped:
            continue
        deduped[g] = []
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            deduped[g].append(idx)
    return deduped

def filter_keyword_matches_in_group(keyword_matches_in_group, deduped_group_docs):
    if not keyword_matches_in_group or not deduped_group_docs:
        return keyword_matches_in_group
    filtered = {}
    for g, kw_map in keyword_matches_in_group.items():
        allowed = set(deduped_group_docs.get(g, []))
        filtered[g] = {}
        for kw, idxs in kw_map.items():
            filtered[g][kw] = [i for i in idxs if i in allowed]
    return filtered

def get_latest_snapshot_for_training():
    snapshot_refine = get_latest_refinement_snapshot()
    if snapshot_refine:
        return snapshot_refine
    snapshot_after = get_latest_training_snapshot("after")
    if snapshot_after:
        return snapshot_after
    snapshot_before = get_latest_training_snapshot("before")
    if snapshot_before:
        return snapshot_before
    return None

def build_keyword_doc_matches_global(group_keywords, df_obj, restrict_groups=None):
    """
    返回每组每个关键词匹配到的文档索引
    restrict_groups: {group: [doc_indices]}，如果提供则只在该组文档内匹配
    """
    keyword_doc_matches = {}
    for grp_name, keywords in (group_keywords or {}).items():
        keyword_doc_matches[grp_name] = {}
        doc_pool = None
        if restrict_groups and grp_name in restrict_groups:
            doc_pool = restrict_groups.get(grp_name, [])
        for kw in keywords:
            cached = get_keyword_doc_indices_cached(kw, df_obj)
            if doc_pool is None:
                matched = list(cached)
            else:
                doc_pool_set = set(doc_pool)
                matched = [i for i in cached if i in doc_pool_set]
            keyword_doc_matches[grp_name][kw] = matched
    return keyword_doc_matches

def get_bm25_cache(df_obj):

    try:
        csv_path = FILE_PATHS["csv_path"]
        csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else None
        if _BM25_CACHE["bm25"] is not None and _BM25_CACHE["csv_mtime"] == csv_mtime:
            return _BM25_CACHE["bm25"], _BM25_CACHE["valid_indices"]
    except Exception:
        pass
    
    from nltk.stem import SnowballStemmer
    from nltk.tokenize import word_tokenize
    import re
    stemmer = SnowballStemmer('english')
    
    def process_articles_serial(articles):
        tokenized_corpus, valid_indices = [], []
        for i, a in enumerate(articles):
            try:
                a = re.sub(r'\d+', '', str(a)).strip()
                if len(a.split()) >= 5:
                    words = word_tokenize(a)
                    stemmed = [stemmer.stem(w.lower()) for w in words]
                    if stemmed:
                        tokenized_corpus.append(" ".join(stemmed))
                        valid_indices.append(i)
            except Exception:
                pass
        return tokenized_corpus, valid_indices
    
    df_clean = df_obj.dropna(subset=[df_obj.columns[1]])
    all_texts = df_clean.iloc[:, 1].astype(str).tolist()
    tokenized_corpus, valid_indices = process_articles_serial(all_texts)
    bm25 = BM25Okapi([s.split() for s in tokenized_corpus])
    
    try:
        csv_path = FILE_PATHS["csv_path"]
        csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else None
        _BM25_CACHE["csv_mtime"] = csv_mtime
    except Exception:
        _BM25_CACHE["csv_mtime"] = None
    _BM25_CACHE["bm25"] = bm25
    _BM25_CACHE["valid_indices"] = valid_indices
    return bm25, valid_indices

def get_keyword_doc_indices_cached(keyword, df_obj):
    try:
        csv_path = FILE_PATHS["csv_path"]
        csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else None
        if _KEYWORD_MATCH_CACHE["csv_mtime"] != csv_mtime:
            _KEYWORD_MATCH_CACHE["csv_mtime"] = csv_mtime
            _KEYWORD_MATCH_CACHE["keyword_to_indices"] = {}
    except Exception:
        pass
    
    key = (keyword or "").strip().lower()
    if not key:
        return []
    cached = _KEYWORD_MATCH_CACHE["keyword_to_indices"].get(key)
    if cached is not None:
        return cached
    
    matches = []
    for i in range(len(df_obj)):
        text = str(df_obj.iloc[i, 1])
        if contains_keyword_word_boundary(text, keyword):
            matches.append(i)
    _KEYWORD_MATCH_CACHE["keyword_to_indices"][key] = matches
    return matches

def get_group_doc_indices_cached(keywords, df_obj):
    group_indices = set()
    for kw in keywords or []:
        for idx in get_keyword_doc_indices_cached(kw, df_obj):
            group_indices.add(idx)
    return sorted(group_indices)

def update_live_keywords_snapshot(group_order):

    try:
        if not group_order:
            return
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        matched_dict_before = {}
        for g, kws in (group_order or {}).items():
            matched_dict_before[g] = get_group_doc_indices_cached(kws, df)

        group_counts_before = {g: len(indices) for g, indices in matched_dict_before.items()}
        keyword_matches_all = build_keyword_doc_matches_global(group_order, df)
        keyword_matches_in_group = build_keyword_doc_matches_global(group_order, df, matched_dict_before)
        keyword_counts = {}
        for group_name, keywords in group_order.items():
            for keyword in keywords:
                keyword_counts[keyword] = len(keyword_matches_all.get(group_name, {}).get(keyword, []))
        
        before_snapshot = {
            "keywords": group_order,
            "keyword_counts": keyword_counts,
            "group_counts": group_counts_before,
            "keyword_matches_all_docs": keyword_matches_all,
            "keyword_matches_in_group": keyword_matches_in_group,
            "group_docs": matched_dict_before
        }
        
        user_data_path = get_user_data_path()
        user_data = load_keysi_user_data()
        if not user_data:
            user_data = {"training_sessions": [], "refinement_changes": []}
        
        user_data["training_sessions"] = [{
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "type": "training",
            "before": before_snapshot
        }]
        user_data["refinement_changes"] = []
        write_json_atomic(user_data_path, user_data)
    except Exception as e:
        pass

def get_all_cls_vectors(df_data, encoder, tokenizer, device):
    vectors = []
    for i in range(len(df_data)):
        text = str(df_data.iloc[i, 1])
        tokens_dict = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        if device != 'cpu':
            tokens_dict = {k: v.to(device) for k, v in tokens_dict.items()}
        
        with torch.no_grad():
            if hasattr(encoder, 'encode_tokens'):
                vector = encoder.encode_tokens(tokens_dict)
            else:
                outputs = encoder(**tokens_dict)
                vector = outputs.last_hidden_state[:, 0, :]
        
        vectors.append(vector.cpu().squeeze(0))
    return torch.stack(vectors, dim=0)
def run_training(skip_gap_filters=False, only_first_stage_training=False):

    gap_warning_text = ""
    try:
        global FINETUNE_LAST_LOSS
        if skip_gap_filters:
            FINETUNE_LAST_LOSS = None
        clear_caches()
        
        df = pd.read_csv(FILE_PATHS["csv_path"])
        all_texts = df.iloc[:,1].fillna("").astype(str).tolist()
        all_labels = df.iloc[:,0].fillna("").astype(str).tolist()
        
    except Exception as e:
        traceback.print_exc()
        return None, None, gap_warning_text

    out_dir = OUTPUT_DIR; os.makedirs(out_dir, exist_ok=True)

    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer('english')
    
    def process_articles_serial(articles):
        tokenized_corpus, valid_indices = [], []
        for i, a in enumerate(articles):
            try:
                a = re.sub(r'\d+', '', str(a)).strip()
                if len(a.split()) >= 5:
                    words = word_tokenize(a)
                    stemmed = [stemmer.stem(w.lower()) for w in words]
                    if stemmed:
                        tokenized_corpus.append(" ".join(stemmed))
                        valid_indices.append(i)
            except Exception:
                pass
        return tokenized_corpus, valid_indices
    
    def custom_tokenizer(text): return text.split()
    def get_main_category(label): return label.split('.')[0] if '.' in label else label
    
    def bm25_search_batch(bm25, query_groups, valid_indices):
        results = {}
        for g, words in query_groups.items():
            q = [stemmer.stem(w.lower()) for w in words]
            scores = bm25.get_scores(q)
            
            idx_corpus = [i for i, s in enumerate(scores) if s > 0.1]  
            if len(idx_corpus) == 0:
                idx_corpus = [i for i, s in enumerate(scores) if s > 0.01]
            
            idx_orig = [valid_indices[i] for i in idx_corpus]
            results[g] = idx_orig[:3000]  

        return results

    tokenized_corpus, valid_indices = process_articles_serial(all_texts)
    bm25 = BM25Okapi([s.split() for s in tokenized_corpus])

    USER_GROUPS_ONLY = True
    ALLOW_EMPTY_GROUPS = False
    
    query_groups = {}
    user_groups_all = {}
    matched_dict = None
    use_snapshot_groups = False
    
    snapshot_for_training = get_latest_snapshot_for_training()
    if snapshot_for_training and snapshot_for_training.get("keywords"):
        user_groups_all = dict(snapshot_for_training.get("keywords"))
        for group_name, keywords in user_groups_all.items():
            if keywords:
                query_groups[group_name] = keywords
    
    if snapshot_for_training and snapshot_for_training.get("group_docs"):
        matched_dict = {g: list(idxs) for g, idxs in snapshot_for_training.get("group_docs").items()}
        use_snapshot_groups = True
    
    if not user_groups_all:
        if os.path.exists(FILE_PATHS["final_list_path"]):
            try:
                with open(FILE_PATHS["final_list_path"], "r", encoding="utf-8") as f:
                    user_groups = json.load(f)
                user_groups_all = dict(user_groups)
                for group_name, keywords in user_groups.items():
                    if keywords:  
                        query_groups[group_name] = keywords
            except Exception as e:
                traceback.print_exc()
                query_groups = {}
        else:
            query_groups = {}
    
    if not query_groups and not use_snapshot_groups and not ALLOW_EMPTY_GROUPS:
        return None, None, gap_warning_text
    
    if not use_snapshot_groups:
        try:
            matched_dict = bm25_search_batch(bm25, query_groups, valid_indices)
        except Exception as e:
            traceback.print_exc()
            return None, None, gap_warning_text
    if "Exclude" in user_groups_all and "Exclude" not in matched_dict:
        matched_dict["Exclude"] = []
    for g in user_groups_all.keys():
        if g not in matched_dict:
            matched_dict[g] = []
    for g, idxs in matched_dict.items():
        pass

    
    with open(FILE_PATHS["bm25_search_results"], "w", encoding="utf-8") as f:
        json.dump(matched_dict, f, ensure_ascii=False, indent=2)


    try:
        tokenizer = BertTokenizer.from_pretrained(LOCKED_BERT_NAME)
    except Exception as e:
        traceback.print_exc()
        return None, None, gap_warning_text
    
    try:
        encoder = SentenceEncoder(device=device)
    except Exception as e:
           
        traceback.print_exc()
        return None, None, gap_warning_text


    class TripletTextDataset(torch.utils.data.Dataset):
        def __init__(self, triplets, texts):
            self.triplets, self.texts = triplets, texts
        def __len__(self): return len(self.triplets)
        def __getitem__(self, i):
            a,p,n = self.triplets[i]
            return self.texts[a], self.texts[p], self.texts[n]

    def collate_triplet(batch, tokenizer, max_len=256, device='cpu'):
        A,P,N = zip(*batch)
        tok = lambda T: tokenizer(list(T), return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        A = {k:v.to(device) for k,v in tok(A).items()}
        P = {k:v.to(device) for k,v in tok(P).items()}
        N = {k:v.to(device) for k,v in tok(N).items()}
        return A,P,N

    def generate_triplets_from_groups(matched_dict, min_pos_per_group=None, num_pos_per_anchor=None, num_neg_per_anchor=None):
        if min_pos_per_group is None: min_pos_per_group = get_config("min_pos_per_group")
        if num_pos_per_anchor is None: num_pos_per_anchor = get_config("num_pos_per_anchor")
        if num_neg_per_anchor is None: num_neg_per_anchor = get_config("num_neg_per_anchor")
        pos_groups = {g:idxs for g,idxs in matched_dict.items() if g!="Exclude" and len(idxs)>=min_pos_per_group}
        if not pos_groups: return []
        
        triplets = []
        for group_name, idxs in pos_groups.items():
            neg_pool = []
            for g, g_idxs in matched_dict.items():
                if g != group_name:  
                    neg_pool.extend(g_idxs)
            
            if len(neg_pool) == 0:  
                continue
            
            for a in idxs:
                pos = [x for x in idxs if x!=a]
                if len(pos) > num_pos_per_anchor: 
                    pos = random.sample(pos, num_pos_per_anchor)
                
             
                neg = random.sample(neg_pool, k=min(num_neg_per_anchor, len(neg_pool)))
                
                for p in pos:
                    for n in neg:
                        if n==a or n==p: continue
                        triplets.append((a,p,n))
        
        random.shuffle(triplets)
        return triplets

    def semi_hard_triplet(za, zp, zn, margin=0.8):

        d_ap = torch.norm(za - zp, dim=1)
        d_an = torch.norm(za - zn, dim=1)
        mask = (d_an > d_ap) & (d_an < d_ap + margin)
        if mask.any():
            return nn.TripletMarginLoss(margin=margin, p=2)(za[mask], zp[mask], zn[mask])
        
      
        with torch.no_grad():
            D = torch.cdist(za, zn, p=2)
            n_hard = D.argmin(dim=1)
        return nn.TripletMarginLoss(margin=margin, p=2)(za, zp, zn[n_hard])

    def train_triplet_text(model, tokenizer, triplets, texts, device, epochs=None, bs=None, margin=None, lr=None, freeze_layers=None):

        if epochs is None: epochs = get_config("triplet_epochs")
        if bs is None: bs = get_config("triplet_batch_size")
        if margin is None: margin = get_config("triplet_margin")
        if lr is None: lr = get_config("triplet_lr")
        if freeze_layers is None: freeze_layers = get_config("freeze_layers")
        global FINETUNE_LAST_LOSS
        if hasattr(model,'bert'):
            for p in model.bert.embeddings.parameters(): p.requires_grad=False
            for L in model.bert.encoder.layer[:freeze_layers]:
                for p in L.parameters(): p.requires_grad=False
        params = [p for p in model.parameters() if p.requires_grad]

        ds = TripletTextDataset(triplets, texts)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                         collate_fn=lambda b: collate_triplet(b, tokenizer, device=device))
        opt = torch.optim.AdamW(params, lr=lr)

        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for step, (A,P,N) in enumerate(dl):
                opt.zero_grad()
                za = model.encode_tokens(A)
                zp = model.encode_tokens(P)
                zn = model.encode_tokens(N)
                loss = semi_hard_triplet(za, zp, zn, margin)
                try:
                    FINETUNE_LAST_LOSS = float(loss.detach().item())
                except Exception:
                    pass
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl)
            print(f"[Triplet] epoch {ep+1}/{epochs} loss={avg_loss:.4f}")
        return model

    def encode_corpus(model, tokenizer, texts, device, bs=None, max_len=None):
        if bs is None: bs = get_config("encoding_batch_size")
        if max_len is None: max_len = get_config("max_length")
        model.eval()
        Z = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                tokens = {k:v.to(device) for k,v in tokens.items()}
                z = model.encode_tokens(tokens)
                Z.append(z.cpu().numpy())
        return np.vstack(Z)

    def gap_based_group_filtering(Z_raw, matched_dict, label_map, alpha=None, query_groups=None, all_texts=None):
        if alpha is None:
            alpha = get_config("gap_alpha")
        warning_groups = []
        Z_np = Z_raw.copy()
        Z_norm = Z_np / (np.linalg.norm(Z_np, axis=1, keepdims=True) + 1e-8)
        group_centers = {}
        tau_ex = get_config("gap_exclude_concentration_threshold")
        for group_name, indices in matched_dict.items():
            if len(indices) == 0:
                continue
            valid_indices = [idx for idx in indices if idx < len(Z_norm)]
            if len(valid_indices) == 0:
                continue
            if query_groups and all_texts and group_name in query_groups:
                keyword_protos = []
                for kw in query_groups[group_name]:
                    matched = [i for i in valid_indices if i < len(all_texts) and contains_keyword_word_boundary(str(all_texts[i]), kw)]
                    if matched:
                        keyword_protos.append(np.mean(Z_norm[matched], axis=0))
                if keyword_protos:
                    mu = np.mean(keyword_protos, axis=0)
                else:
                    mu = np.mean(Z_norm[valid_indices], axis=0)
            else:
                mu = np.mean(Z_norm[valid_indices], axis=0)
            if group_name == "Exclude":
                c_ex = np.linalg.norm(mu)
                if c_ex < tau_ex:
                    continue
            group_center = mu / (np.linalg.norm(mu) + 1e-8)
            group_centers[group_name] = group_center
        
        all_similarities = []
        for group_name, center in group_centers.items():
            sim = np.dot(Z_norm, center)
            all_similarities.append(sim)
        
        if len(all_similarities) == 0:

            return matched_dict, warning_groups
        
        all_similarities = np.array(all_similarities).T
        doc_pool = sorted({idx for idxs in matched_dict.values() for idx in idxs if 0 <= idx < len(Z_norm)})
        if not doc_pool:
            return matched_dict, warning_groups
        all_similarities = all_similarities[doc_pool]
        
        sorted_indices = np.argsort(all_similarities, axis=1)[:, ::-1]  
        s_top1 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 0]]
        s_top2 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 1]] if all_similarities.shape[1] > 1 else s_top1
        gap = s_top1 - s_top2
        
        group_names = list(group_centers.keys())
        arg1 = sorted_indices[:, 0]  
        
        
        group_thresholds = {}
        
        for group_name, group_idx in label_map.items():
            if group_name not in group_centers:
                continue
                

            if group_name in group_names:
                group_center_idx = group_names.index(group_name)
                
                group_mask = (arg1 == group_center_idx)
                if not group_mask.any():
                    continue
                    
                gaps_group = gap[group_mask]
                mean_gap = gaps_group.mean()
                std_gap = gaps_group.std()
                

                base_threshold = mean_gap - alpha * std_gap
                
                min_samples = get_config("gap_min_samples")
                percentile_fallback = get_config("gap_percentile_fallback")
                if len(gaps_group) < min_samples or std_gap < 1e-6:

                    threshold = np.percentile(gaps_group, percentile_fallback)
                else:

                    global_median = np.median(gap)
                    thr_floor = get_config("gap_floor_threshold")
                    mix_ratio = get_config("gap_mix_ratio")
                    threshold = max((1 - mix_ratio) * base_threshold + mix_ratio * global_median, thr_floor)
                
                group_thresholds[group_name] = threshold
                
        
        keep_mask = np.ones(len(gap), dtype=bool)
        filtered_by_group = {}
        
        for i, gap_val in enumerate(gap):
            if arg1[i] < len(group_names):
                group_name = group_names[arg1[i]]
                if group_name == "Exclude":
                    continue
                if group_name in group_thresholds:
                    if gap_val < group_thresholds[group_name]:
                        keep_mask[i] = False
                        if group_name not in filtered_by_group:
                            filtered_by_group[group_name] = 0
                        filtered_by_group[group_name] += 1
        
        filtered_count = np.sum(~keep_mask)
        for group_name, count in filtered_by_group.items():
            pass
        clean_matched_dict = {}
        for group_name, indices in matched_dict.items():
            if group_name == "Exclude":
                clean_matched_dict[group_name] = indices  
                continue
                
            if group_name in group_centers:
                group_center_idx = group_names.index(group_name)
                orig_idxs = set(indices)
                group_mask = (arg1 == group_center_idx) & keep_mask
                kept_doc_pool = [doc_pool[i] for i in np.where(group_mask)[0]]
                filtered_indices = [int(i) for i in kept_doc_pool if int(i) in orig_idxs]
                if len(indices) > 0 and len(filtered_indices) == 0:
                    filtered_indices = list(indices)
                    warning_groups.append(group_name)
                clean_matched_dict[group_name] = filtered_indices
            else:
                clean_matched_dict[group_name] = indices
        
        for group_name, indices in clean_matched_dict.items():
            pass
        return clean_matched_dict, warning_groups

    def build_group_prototypes(encoder, tokenizer, texts, matched_dict, device, bs=None, min_per_group=None, query_groups=None):
        if bs is None: bs = get_config("encoding_batch_size")
        if min_per_group is None: min_per_group = get_config("min_per_group_prototype")

        def encode_all():
            encoder.eval()
            Z=[]
            for i in range(0, len(texts), bs):
                toks = tokenizer(texts[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k:v.to(device) for k,v in toks.items()}
                Z.append(encoder.encode_tokens(toks).detach().cpu())
            return torch.vstack(Z)  
        
        Z_all = encode_all()           
        G = {}
        
        if query_groups:
            for g, idxs in matched_dict.items():
                if g == "Exclude" or g not in query_groups:
                    continue
                
                group_keywords = query_groups[g]
                if not group_keywords:
                    continue
                
                idxs = [i for i in idxs if 0 <= i < len(Z_all)]
                if len(idxs) < min_per_group:
                    continue
                
                keyword_prototypes = []
                for keyword in group_keywords:
                    keyword_lower = keyword.lower()
                    keyword_matched_docs = []
                    
                    for doc_idx in idxs:
                        if doc_idx < len(texts):
                            doc_text = str(texts[doc_idx])
                            if contains_keyword_word_boundary(doc_text, keyword):
                                keyword_matched_docs.append(doc_idx)
                    
                    if len(keyword_matched_docs) > 0:
                        keyword_embeddings = Z_all[keyword_matched_docs]
                        keyword_proto = keyword_embeddings.mean(0)
                        keyword_prototypes.append(keyword_proto)
                    else:
                        pass
                
                if len(keyword_prototypes) > 0:
                    proto = torch.stack(keyword_prototypes).mean(0)
                    G[g] = nn.functional.normalize(proto, dim=0)  
                else:
                    proto = Z_all[idxs].mean(0)
                    G[g] = nn.functional.normalize(proto, dim=0)
        else:
            for g, idxs in matched_dict.items():
                if g == "Other": 
                    continue
                idxs = [i for i in idxs if 0 <= i < len(Z_all)]
                if len(idxs) >= min_per_group:
                    proto = Z_all[idxs].mean(0)
                    G[g] = nn.functional.normalize(proto, dim=0)  
        
        return G  

    def prototype_center_training(encoder, tokenizer, all_texts, group_prototypes, device,
                                 epochs=None, bs=None, lr=None, matched_dict=None):

        if epochs is None: epochs = get_config("proto_epochs")
        if bs is None: bs = get_config("proto_batch_size")
        if lr is None: lr = get_config("proto_lr")
        global FINETUNE_LAST_LOSS

        if hasattr(encoder, 'bert'):
            for p in encoder.bert.embeddings.parameters(): 
                p.requires_grad = False
            for layer in encoder.bert.encoder.layer[:4]:
                for p in layer.parameters(): 
                    p.requires_grad = False
        
        global_prototypes = {}
        for group_name, proto_tensor in group_prototypes.items():
            if group_name != "Other":
                global_prototypes[group_name] = proto_tensor.clone().to(device).detach()
        
        params = [p for p in encoder.parameters() if p.requires_grad]
        
        opt = torch.optim.AdamW(params, lr=lr)
        
        class BalancedBatchSampler:
            def __init__(self, indices_by_group, groups, m_per_group=4, batch_size=64):
                self.buckets = {g: list(idxs) for g, idxs in indices_by_group.items()}
                self.groups = [g for g in groups if len(self.buckets[g]) >= m_per_group]
                self.m = m_per_group
                self.batch_size = batch_size
                
            def __iter__(self):
                groups_shuffled = self.groups.copy()
                random.shuffle(groups_shuffled)
                
                for i in range(0, len(groups_shuffled), 2):
                    gs = groups_shuffled[i:i+2]
                    if len(gs) < 2:
                        break
                    
                    batch = []
                    for g in gs:
                        group_samples = self.buckets[g].copy()
                        random.shuffle(group_samples)
                        batch.extend(group_samples[:self.m])
                    
                    if len(batch) < self.batch_size:
                        remaining = self.batch_size - len(batch)
                        all_samples = []
                        for g, samples in self.buckets.items():
                            if g not in gs:  
                                all_samples.extend(samples)
                        if all_samples:
                            random.shuffle(all_samples)
                            batch.extend(all_samples[:remaining])
                    
                    yield batch[:self.batch_size]
                    
            def __len__(self):
                return max(1, len(self.groups) // 2)
        
        m_per_group = bs // 4
        sampler = BalancedBatchSampler(matched_dict, list(global_prototypes.keys()), 
                                     m_per_group=m_per_group, batch_size=bs)
        
        print(f"Balanced Batch: {list(global_prototypes.keys())}")
        print(f"Balanced Batch: {m_per_group}")
        
        encoder.train()
        for ep in range(epochs):
            total_center_loss = 0
            total_loss = 0
            steps = 0
            
            for batch_indices in sampler:
                batch_texts = [all_texts[i] for i in batch_indices]
                
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                                 truncation=True, max_length=256).to(device)
                outputs = encoder.encode_tokens(inputs)
                z_batch = nn.functional.normalize(outputs, p=2, dim=-1)
                
                batch_group_means = {}
                for group_name, indices in matched_dict.items():
                    if group_name == "Exclude":
                        continue
                    
                    group_mask = torch.tensor([i in indices for i in batch_indices], device=device)
                    if group_mask.sum() > 0:
                        group_embeddings = z_batch[group_mask]
                        batch_group_means[group_name] = group_embeddings.mean(dim=0)
                
                
                center_loss = torch.tensor(0.0, device=device)
                for group_name, indices in matched_dict.items():
                    if group_name == "Exclude" or group_name not in global_prototypes:
                        continue
                    
                    group_mask = torch.tensor([i in indices for i in batch_indices], device=device)
                    if group_mask.sum() > 0:
                        group_embeddings = z_batch[group_mask]
                        prototype = global_prototypes[group_name]
                        
                        distances = torch.norm(group_embeddings - prototype, p=2, dim=1)
                        center_loss += distances.mean()
                
                total_batch_loss = center_loss
                
                if total_batch_loss > 0:
                    opt.zero_grad()
                    try:
                        FINETUNE_LAST_LOSS = float(center_loss.detach().item())
                    except Exception:
                        pass
                    total_batch_loss.backward()
                    opt.step()
                    
                    
                    total_center_loss += center_loss.item()
                    total_loss += total_batch_loss.item()
                    steps += 1
            
            if steps > 0:
                avg_center = total_center_loss / steps
                avg_total = total_loss / steps
                print(f"[Proto] epoch {ep+1}/{epochs} center_loss={avg_center:.4f} total={avg_total:.4f}")
            print(f"  Epoch {ep+1}...")

            with torch.no_grad():
                encoder.eval()
                Z_all = []
                for i in range(0, len(all_texts), bs):
                    batch_texts = all_texts[i:i+bs]
                    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                                     truncation=True, max_length=256).to(device)
                    outputs = encoder.encode_tokens(inputs)
                    Z_all.append(outputs.detach())
                Z_all = torch.cat(Z_all, dim=0) 
                Z_all = nn.functional.normalize(Z_all, p=2, dim=-1)
                
                ema_alpha = get_config("ema_alpha")  
                for group_name, indices in matched_dict.items():
                    if group_name != "Exclude" and group_name in global_prototypes:
                        valid_indices = [i for i in indices if 0 <= i < len(Z_all)]
                        if len(valid_indices) > 0:

                            current_proto = Z_all[valid_indices].mean(dim=0)
                            current_proto = nn.functional.normalize(current_proto, p=2, dim=-1)
                            
                            global_prototypes[group_name] = (1 - ema_alpha) * global_prototypes[group_name] + ema_alpha * current_proto
                            global_prototypes[group_name] = nn.functional.normalize(global_prototypes[group_name], p=2, dim=-1)
                
                encoder.train() 
        
        for group_name, indices in matched_dict.items():
            print(f"  {group_name}: {len(indices)} docs")
        return encoder

    def evaluate_clustering_quality(Z, true_labels, matched_dict, group_names):
        
       
        
        silhouette = silhouette_score(Z, true_labels)
        
        group_labels = []
        for i in range(len(true_labels)):
            assigned = False
            for group_name in group_names:
                if i in matched_dict.get(group_name, []):
                    group_labels.append(group_name)
                    assigned = True
                    break
            if not assigned:
                group_labels.append("Unassigned")
        
        group_silhouette = silhouette_score(Z, group_labels)
        
        group_stats = {}
        for group_name in group_names:
            if group_name in matched_dict:
                group_indices = matched_dict[group_name]
                if len(group_indices) > 1:
                    group_embeddings = Z[group_indices]
                    
                    intra_distances = []
                    for i in range(len(group_embeddings)):
                        for j in range(i+1, len(group_embeddings)):
                            dist = np.linalg.norm(group_embeddings[i] - group_embeddings[j])
                            intra_distances.append(dist)
                    intra_cohesion = np.mean(intra_distances) if intra_distances else 0
                    
                    other_indices = []
                    for other_group in group_names:
                        if other_group != group_name and other_group in matched_dict:
                            other_indices.extend(matched_dict[other_group])
                    
                    if other_indices:
                        group_center = np.mean(group_embeddings, axis=0)
                        other_embeddings = Z[other_indices]
                        inter_distances = [np.linalg.norm(group_center - other_emb) for other_emb in other_embeddings]
                        inter_separation = np.mean(inter_distances)
                    else:
                        inter_separation = 0
                    
                    separation_ratio = inter_separation / intra_cohesion if intra_cohesion > 0 else 0
                    
                    group_stats[group_name] = {
                        'size': len(group_indices),
                        'intra_cohesion': float(intra_cohesion),
                        'inter_separation': float(inter_separation),
                        'separation_ratio': float(separation_ratio)
                    }
        
        return {
            'silhouette_true_labels': silhouette,
            'silhouette_group_labels': group_silhouette,
            'group_stats': group_stats
        }

    

    for group_name, indices in matched_dict.items():
        if group_name == "Exclude":
            continue
        
        if len(indices) == 0:
            continue
            
        group_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        if len(group_labels) == 0:
            continue
            
        from collections import Counter
        label_counts = Counter(group_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        purity = most_common_count / len(group_labels)
        
    

    import copy
    pre_dedupe_matched_dict = copy.deepcopy(matched_dict)
    doc_to_group = {}  
    final_matched_dict = {}
    
    ordered_groups = list(matched_dict.keys())
    if "Exclude" in ordered_groups:
        ordered_groups = ["Exclude"] + [g for g in ordered_groups if g != "Exclude"]
    for group_name in ordered_groups:
        indices = matched_dict.get(group_name, [])
        if group_name == "Exclude":
            final_matched_dict[group_name] = []
            for doc_id in indices:
                if doc_id not in doc_to_group:
                    doc_to_group[doc_id] = group_name
                    final_matched_dict[group_name].append(doc_id)
                else:
                    pass
            continue
            
        final_matched_dict[group_name] = []
        for doc_id in indices:
            if doc_id not in doc_to_group:
                doc_to_group[doc_id] = group_name
                final_matched_dict[group_name].append(doc_id)
            else:
                pass

    matched_dict = final_matched_dict
    dedupe_removed_by_group = {}
    for g in pre_dedupe_matched_dict.keys():
        before_set = set(pre_dedupe_matched_dict.get(g, []))
        after_set = set(matched_dict.get(g, []))
        dedupe_removed_by_group[g] = sorted(before_set - after_set)
    

    for group_name, indices in matched_dict.items():
        pass

    Z_raw = encode_corpus(encoder, tokenizer, all_texts, device)
    
    label_map, cur = {}, 0
    for g in ["Group 1", "Group 2", "Group 3", "Exclude"]:
        if g in matched_dict:
            label_map[g] = cur; cur += 1
    
    bm25_results = copy.deepcopy(matched_dict)
    clean_matched_dict = matched_dict

    user_data = load_keysi_user_data()
    training_sessions = user_data.get("training_sessions", [])
    gap_results_path = FILE_PATHS.get("gap_based_filter_results")
    deleted_gap_file = False
    if gap_results_path and os.path.exists(gap_results_path) and not skip_gap_filters:
        try:
            os.remove(gap_results_path)
            deleted_gap_file = True
        except Exception:
            pass

    gap1_removed_by_group = {g: [] for g in bm25_results.keys()}
    removed_by_group = {g: [] for g in bm25_results.keys()}

    if skip_gap_filters:
        clean_matched_dict_gap1 = matched_dict
        gap1_warning_groups = []
        gap_warning_text = ""
    else:
        clean_matched_dict_gap1, gap1_warning_groups = gap_based_group_filtering(
            Z_raw,
            matched_dict,
            label_map,
            alpha=0.7,
            query_groups=query_groups,
            all_texts=all_texts,
        )
        if gap1_warning_groups:
            gap_warning_text = "Semantic gap too large for group(s) in gap1: " + ", ".join(gap1_warning_groups) + ". Original documents were kept."
        user_data["gap_filter_applied_once"] = True

        try:
            if training_sessions:
                latest_session = training_sessions[-1]
                removed_by_group = {}
                gap1_removed_by_group = {}
                for g in bm25_results.keys():
                    if g == "Exclude":
                        gap1_removed_by_group[g] = []
                        removed_by_group[g] = []
                        continue
                    before_set = set(bm25_results.get(g, []))
                    after_set = set(clean_matched_dict_gap1.get(g, []))
                    removed = sorted(before_set - after_set)
                    gap1_removed_by_group[g] = removed
                    removed_by_group[g] = removed
                latest_session["gap_filter"] = {
                    "applied": True,
                    "deleted_gap_results_file": deleted_gap_file,
                    "removed_doc_indices_by_group": removed_by_group,
                    "dedupe_removed_by_group": dedupe_removed_by_group,
                    "gap1": {
                        "removed_doc_indices_by_group": gap1_removed_by_group
                    }
                }
                user_data_path = get_user_data_path()
                write_json_atomic(user_data_path, user_data)
            else:
                user_data_path = get_user_data_path()
                write_json_atomic(user_data_path, user_data)
        except Exception:
            pass

    matched_dict_for_display = copy.deepcopy(matched_dict)
    matched_dict = clean_matched_dict_gap1
    

    try:
        filtered_path = FILE_PATHS.get("filtered_group_assignment")
        if filtered_path and os.path.exists(filtered_path):
            os.remove(filtered_path)
    except Exception as e:
        pass


    def train_triplet_and_center(current_matched_dict, stage_label):
        group_prototypes = build_group_prototypes(
            encoder, tokenizer, all_texts, current_matched_dict, device, query_groups=query_groups
        )
        for group_name, indices in current_matched_dict.items():
            if group_name == "Other":
                continue
            if len(indices) == 0:
                continue
            group_labels = [all_labels[i] for i in indices if i < len(all_labels)]
            if len(group_labels) == 0:
                continue
            label_counts = Counter(group_labels)
            most_common_label, most_common_count = label_counts.most_common(1)[0]
            purity = most_common_count / len(group_labels)
        for g, idxs in current_matched_dict.items():
            pass
        try:
            triplets = generate_triplets_from_groups(current_matched_dict)
            triplets_count = len(triplets)
            if len(triplets) > 0:
                enc = train_triplet_text(encoder, tokenizer, triplets, all_texts, device)
            else:
                enc = encoder
        except Exception as e:
            traceback.print_exc()
            return None
        try:
            enc = prototype_center_training(enc, tokenizer, all_texts, group_prototypes, device,
                                            matched_dict=current_matched_dict)
        except Exception as e:
            traceback.print_exc()
            return None
        return enc, triplets_count

    train_result = train_triplet_and_center(matched_dict, "gap1")
    if train_result is None:
        return None, None, gap_warning_text
    encoder, triplets_count_gap1 = train_result

    try:
        Z_trained_gap2_source = encode_corpus(encoder, tokenizer, all_texts, device)
    except Exception as e:
        traceback.print_exc()
        return None, None, gap_warning_text

    matched_dict_gap2_source = copy.deepcopy(matched_dict_for_display)
    label_map2, cur2 = {}, 0
    for g in ["Group 1", "Group 2", "Group 3", "Exclude"]:
        if g in matched_dict_gap2_source:
            label_map2[g] = cur2
            cur2 += 1
    if skip_gap_filters:
        clean_matched_dict_gap2 = matched_dict_gap2_source
        gap2_warning_groups = []
    else:
        clean_matched_dict_gap2, gap2_warning_groups = gap_based_group_filtering(
            Z_trained_gap2_source,
            matched_dict_gap2_source,
            label_map2,
            alpha=0.5,
            query_groups=query_groups,
            all_texts=all_texts
        )
        if gap2_warning_groups:
            suffix = "Semantic gap too large for group(s) in gap2: " + ", ".join(gap2_warning_groups) + ". Original documents were kept."
            gap_warning_text = f"{gap_warning_text} {suffix}".strip() if gap_warning_text else suffix

    gap2_removed_by_group = {}
    for g in matched_dict_gap2_source.keys():
        if g == "Exclude":
            gap2_removed_by_group[g] = []
            continue
        before_set = set(matched_dict_gap2_source.get(g, []))
        after_set = set(clean_matched_dict_gap2.get(g, []))
        gap2_removed_by_group[g] = sorted(before_set - after_set)

    matched_dict = clean_matched_dict_gap2
    matched_dict_for_display = copy.deepcopy(matched_dict)

    if not skip_gap_filters:
        try:
            user_data = load_keysi_user_data()
            training_sessions = user_data.get("training_sessions", [])
            if training_sessions:
                latest_session = training_sessions[-1]
                gap_filter_info = latest_session.get("gap_filter", {})
                gap_filter_info["gap2"] = {"removed_doc_indices_by_group": gap2_removed_by_group}
                gap_filter_info["removed_doc_indices_by_group"] = gap2_removed_by_group
                latest_session["gap_filter"] = gap_filter_info
                user_data_path = get_user_data_path()
                write_json_atomic(user_data_path, user_data)
        except Exception:
            pass

    try:
        record_user_data("training_after", group_order=user_groups_all, matched_dict=matched_dict)
    except Exception as e:
        pass

    try:
        def _group_purity(indices, labels):
            if not indices or not labels:
                return None, None, None
            group_labels = [labels[i] for i in indices if i < len(labels)]
            if not group_labels:
                return None, None, None
            cnt = Counter(group_labels)
            top_label, top_count = cnt.most_common(1)[0]
            return round(top_count / len(group_labels), 4), top_label, top_count

        safe_name = get_safe_user_name()
        gapresult_path = os.path.join(OUTPUT_DIR, f"{safe_name}_gapresult.json")
        report = {
            "username": safe_name,
            "initial_match": {},
            "after_gap1": {},
            "after_gap2": {},
        }
        for g in bm25_results.keys():
            init_idxs = sorted(bm25_results.get(g, []))
            n0 = len(init_idxs)
            p0, dom0, nc0 = _group_purity(init_idxs, all_labels)
            report["initial_match"][g] = {
                "matched_doc_indices": init_idxs,
                "count": n0,
                "accuracy_purity": p0,
                "dominant_label": dom0,
                "dominant_count": nc0,
            }
            rem1 = list(gap1_removed_by_group.get(g, []))
            kept1 = sorted(clean_matched_dict_gap1.get(g, []))
            k1, n_rem1 = len(kept1), len(rem1)
            p1, dom1, n1 = _group_purity(kept1, all_labels)
            coverage_gap1 = round(k1 / n0, 4) if n0 else None
            loss_rate_gap1 = round(n_rem1 / n0, 4) if n0 else None
            report["after_gap1"][g] = {
                "gap1_removed_doc_indices": rem1,
                "gap1_removed_count": n_rem1,
                "kept_doc_indices": kept1,
                "kept_count": k1,
                "coverage_vs_initial": coverage_gap1,
                "loss_rate_vs_initial": loss_rate_gap1,
                "accuracy_purity_after_gap1": p1,
                "dominant_label": dom1,
                "dominant_count": n1,
            }
            rem2 = list(gap2_removed_by_group.get(g, []))
            kept2 = sorted(clean_matched_dict_gap2.get(g, []))
            k2, n_rem2 = len(kept2), len(rem2)
            p2, dom2, n2 = _group_purity(kept2, all_labels)
            coverage_gap2 = round(k2 / n0, 4) if n0 else None
            loss_rate_gap2 = round((n0 - k2) / n0, 4) if n0 else None
            loss_rate_gap2_only = round(n_rem2 / k1, 4) if k1 else None
            report["after_gap2"][g] = {
                "gap2_removed_doc_indices": rem2,
                "gap2_removed_count": n_rem2,
                "kept_doc_indices": kept2,
                "kept_count": k2,
                "coverage_vs_initial": coverage_gap2,
                "loss_rate_vs_initial": loss_rate_gap2,
                "loss_rate_gap2_only": loss_rate_gap2_only,
                "accuracy_purity_after_gap2": p2,
                "dominant_label": dom2,
                "dominant_count": n2,
            }
        if write_json_atomic(gapresult_path, report):
            pass
        else:
            pass
    except Exception as e:
        traceback.print_exc()

    if only_first_stage_training:
        # refinement mode: 只跑 gap1 阶段的 triplet+center，再直接进入后处理
        triplets_count_gap2 = triplets_count_gap1
    else:
        train_result = train_triplet_and_center(matched_dict, "gap2")
        if train_result is None:
            return None, None, gap_warning_text
        encoder, triplets_count_gap2 = train_result
    
    
    def l2norm(X): 
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    try:
        Z_current = encode_corpus(encoder, tokenizer, all_texts, device)
        Zn = l2norm(Z_current)
    except Exception as e:
        traceback.print_exc()
        return None, None, gap_warning_text
    
    group_stats = {}
    ema_alpha = get_config("ema_alpha")  
    
    for g, train_idxs in matched_dict.items():
        if g == "Other" or len(train_idxs) < 5: 
            continue
        train_idxs = [i for i in train_idxs if 0 <= i < len(Zn)]
        if len(train_idxs) < 5: 
            continue

  
        proto = Zn[train_idxs].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        

        d_train = np.linalg.norm(Zn[train_idxs] - proto, axis=1)
        r_core = np.quantile(d_train, 0.50)
        r_near = np.quantile(d_train, 0.80)
        r_edge = np.quantile(d_train, 0.90)
        
        group_stats[g] = {
            "proto": proto, 
            "train_idxs": train_idxs, 
            "r_core": float(r_core), 
            "r_near": float(r_near), 
            "r_edge": float(r_edge),
            "ema_proto": proto.copy(),  
            "ema_r_core": float(r_core),  
            "ema_r_near": float(r_near),
            "ema_r_edge": float(r_edge)
        }
    
    
    for g, stat in group_stats.items():
        train_set = set(stat["train_idxs"])
        proto = stat["ema_proto"]
        r_core = stat["ema_r_core"]
        
        d_all = np.linalg.norm(Zn - proto, axis=1)
        
        high_conf_new = []
        for i, d in enumerate(d_all):
            if i not in train_set and d <= r_core:
                high_conf_new.append((i, d))
        
        if len(high_conf_new) > 0:
            
            high_conf_indices = [i for i, _ in high_conf_new]
            high_conf_embeddings = Zn[high_conf_indices]
            
            new_proto = high_conf_embeddings.mean(axis=0)
            new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-8)
            
            stat["ema_proto"] = (1 - ema_alpha) * stat["ema_proto"] + ema_alpha * new_proto
            stat["ema_proto"] = stat["ema_proto"] / (np.linalg.norm(stat["ema_proto"]) + 1e-8)
            
            all_relevant_indices = stat["train_idxs"] + high_conf_indices
            d_combined = np.linalg.norm(Zn[all_relevant_indices] - stat["ema_proto"], axis=1)
            
            new_r_core = np.quantile(d_combined, 0.50)
            new_r_near = np.quantile(d_combined, 0.80) 
            new_r_edge = np.quantile(d_combined, 0.90)
            
            stat["ema_r_core"] = (1 - ema_alpha) * stat["ema_r_core"] + ema_alpha * new_r_core
            stat["ema_r_near"] = (1 - ema_alpha) * stat["ema_r_near"] + ema_alpha * new_r_near
            stat["ema_r_edge"] = (1 - ema_alpha) * stat["ema_r_edge"] + ema_alpha * new_r_edge
            
        else:
            pass

    try:
        Z_trained = encode_corpus(encoder, tokenizer, all_texts, device)
    except Exception as e:
        traceback.print_exc()
        return None, None, gap_warning_text

    main_categories = [get_main_category(label) for label in all_labels]
    

    cluster_eval_raw = evaluate_clustering_quality(Z_raw, main_categories, bm25_results, list(bm25_results.keys()))
    
    cluster_eval_trained = evaluate_clustering_quality(Z_trained, main_categories, matched_dict, list(matched_dict.keys()))

    def convert_numpy_types(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    cluster_eval_raw_serializable = convert_numpy_types(cluster_eval_raw)
    cluster_eval_trained_serializable = convert_numpy_types(cluster_eval_trained)
    
    with open(FILE_PATHS["clustering_evaluation"], "w", encoding="utf-8") as f:
        json.dump({
            "raw_embedding": cluster_eval_raw_serializable,
            "trained_embedding": cluster_eval_trained_serializable,
            "triplets_count": triplets_count_gap2,
            "training_epochs": 5,
            "margin": 0.8
        }, f, ensure_ascii=False, indent=2)


    try:

        n = len(Z_raw)
        perp = max(2, min(30, (n - 1) // 3))
        
 
        X2_raw = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            max_iter=get_config("tsne_max_iter"),
            verbose=0
        ).fit_transform(Z_raw)
        
        X2_trained = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            max_iter=get_config("tsne_max_iter"),
            verbose=0
        ).fit_transform(Z_trained)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        unique_main_cats = sorted(list(set(main_categories)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_main_cats)))
        color_map = dict(zip(unique_main_cats, colors))
        
        for main_cat in unique_main_cats:
            cat_mask = [cat == main_cat for cat in main_categories]
            cat_indices = np.where(cat_mask)[0]
            
            if len(cat_indices) > 0:
                cat_2d = X2_raw[cat_indices]
                ax1.scatter(cat_2d[:,0], cat_2d[:,1], 
                           c=[color_map[main_cat]], alpha=0.8, s=60, 
                           marker='o', label=f'{main_cat}')
        
        ax1.set_title(f"BERT Embedding (Silhouette: {cluster_eval_raw['silhouette_true_labels']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1'); ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        for main_cat in unique_main_cats:
            cat_mask = [cat == main_cat for cat in main_categories]
            cat_indices = np.where(cat_mask)[0]
            
            if len(cat_indices) > 0:
                cat_2d = X2_trained[cat_indices]
                ax2.scatter(cat_2d[:,0], cat_2d[:,1], 
                           c=[color_map[main_cat]], alpha=0.8, s=60, 
                           marker='o', label=f'{main_cat}')
        
        ax2.set_title(f"Triplet Embedding (Silhouette: {cluster_eval_trained['silhouette_true_labels']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1'); ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FILE_PATHS["triplet_training_comparison"], dpi=300, bbox_inches='tight')
        
        with open(FILE_PATHS["triplet_tsne_comparison"], "w", encoding="utf-8") as f:
            json.dump({
                "raw_tsne": {
                    "x": X2_raw[:,0].tolist(),
                    "y": X2_raw[:,1].tolist()
                },
                "trained_tsne": {
                    "x": X2_trained[:,0].tolist(),
                    "y": X2_trained[:,1].tolist()
                },
                "labels": all_labels,
                "main_categories": main_categories,
                "silhouette_scores": {
                    "raw": cluster_eval_raw['silhouette_true_labels'],
                    "trained": cluster_eval_trained['silhouette_true_labels']
                }
            }, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        pass

    try:
        np.save(FILE_PATHS["embeddings_trained"], Z_trained)
        write_state_dict_atomic(FILE_PATHS["triplet_trained_encoder"], encoder.state_dict())
        try:
            safe_name = get_safe_user_name()
            model_dir = get_user_model_dir()
            training_model_path = os.path.join(model_dir, f"{safe_name}_keysitraining_model.pth")
            write_state_dict_atomic(training_model_path, encoder.state_dict())
            save_user_data_to_user_dir()
        except Exception as e:
            pass
    except Exception as e:
           
        traceback.print_exc()

    with open(FILE_PATHS["triplet_run_stats"], "w", encoding="utf-8") as f:
        json.dump({
            "bm25_sizes": {k: len(v) for k, v in matched_dict.items()},
            "device": device,
            "proj_dim": encoder.out_dim,
            "triplets_count": triplets_count_gap2,
            "training_epochs": 5,
            "margin": 0.8,
            "silhouette_improvement": float(cluster_eval_trained['silhouette_true_labels'] - cluster_eval_raw['silhouette_true_labels'])
        }, f, ensure_ascii=False, indent=2)
    

    
    
    
    df_articles = df  
    
    encoder_original = SentenceEncoder(device=device)
    encoder_original.eval()
    
    cls_vectors_before = get_all_cls_vectors(df_articles, encoder_original, tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, encoder, tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    
    perplexity_before = min(30, max(5, len(cls_vectors_before) // 3))
    perplexity_after = min(30, max(5, len(cls_vectors_after_cpu) // 3))
    
    tsne_before = TSNE(
        n_components=2,
        perplexity=perplexity_before,
        random_state=42,
        max_iter=get_config("tsne_max_iter"),
        verbose=0
    )
    tsne_after = TSNE(
        n_components=2,
        perplexity=perplexity_after,
        random_state=42,
        max_iter=get_config("tsne_max_iter"),
        verbose=0
    )
    projected_2d_before = tsne_before.fit_transform(cls_vectors_before.numpy())
    projected_2d_after = tsne_after.fit_transform(cls_vectors_after_cpu)
    
    
    group_centers = {}
    
    for group_name, indices in matched_dict.items():
        if len(indices) > 0:
            

            valid_indices = [i for i in indices if i < len(projected_2d_after)]
            
            if len(valid_indices) > 0:
                group_2d_points = projected_2d_after[valid_indices]
                group_center_2d = np.mean(group_2d_points, axis=0)
                group_centers[group_name] = group_center_2d
            else:
                pass
        else:
            pass
    
    def create_plotly_figure(projected_2d, title, is_after=False, highlighted_indices=None, group_centers=None, matched_dict_param=None):
        fig = go.Figure()
        

        article_indices = list(range(len(projected_2d)))
        hover_texts = [f"Article {idx}" for idx in article_indices]
        custom_data = [[idx] for idx in article_indices]
        
        bg_style = PLOT_STYLES["background"]
        fig.add_trace(go.Scatter(
            x=projected_2d[:, 0],
            y=projected_2d[:, 1],
            mode='markers',
            name="All Documents",
            marker=dict(
                color=bg_style["color"],
                size=bg_style["size"],
                opacity=bg_style["opacity"],
                symbol="circle",  
                line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
            ),
            customdata=custom_data,
            hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            hovertext=hover_texts
        ))
        

        if highlighted_indices:

            all_x = projected_2d[:, 0]
            all_y = projected_2d[:, 1]
            
            highlighted_x = [all_x[i] for i in highlighted_indices if i < len(all_x)]
            highlighted_y = [all_y[i] for i in highlighted_indices if i < len(all_y)]
            
            if highlighted_x and highlighted_y:
                core_style = PLOT_STYLES["core"]
                fig.add_trace(go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode='markers',
                    name="Highlighted",
                    marker=dict(
                        color=core_style["color"],
                        size=core_style["size"],
                        symbol=core_style["symbol"],
                        line=dict(width=core_style["line_width"], color=core_style["line_color"])
                    ),
                    customdata=[[i] for i in highlighted_indices if i < len(all_x)],
                    hovertemplate='<b>Article %{customdata[0]}</b><extra></extra>'
                ))
        

        if is_after and group_centers:
            center_style = PLOT_STYLES["center"]
            for group_name, center_2d in group_centers.items():
                color = get_group_color(group_name)
                fig.add_trace(go.Scatter(
                    x=[center_2d[0]],
                    y=[center_2d[1]],
                    mode='markers+text',
                    name=f'Center: {group_name}',
                    marker=dict(
                        color=color,
                        size=center_style["size"],
                        symbol=center_style["symbol"],
                        opacity=center_style["opacity"],
                        line=dict(width=center_style["line_width"], color=center_style["line_color"])
                    ),
                    text=[group_name],
                    textposition="top center",
                    textfont=dict(size=12, color=color, family='Arial Black'),
                    hovertemplate=f'<b>Group Center: {group_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
        else:
            pass

        layout_style = PLOT_STYLES["layout"]
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            showlegend=True,
            hovermode='closest',
            plot_bgcolor=layout_style["plot_bgcolor"],
            paper_bgcolor=layout_style["paper_bgcolor"],
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(**layout_style["xaxis"], title="X"),
            yaxis=dict(**layout_style["yaxis"], title="Y")
        )
        
        return fig

    fig_before = create_plotly_figure(projected_2d_before, "2D Projection Before Finetuning", False, None, None, None)
    fig_after = create_plotly_figure(projected_2d_after, "2D Projection After Finetuning", True, None, group_centers, matched_dict_for_display)
    

    
    return fig_before, fig_after, gap_warning_text


def run_training_with_highlights(highlighted_indices):

    global df
    
    if isinstance(highlighted_indices, tuple):
        highlighted_indices = list(highlighted_indices)
    
    encoder_original = SentenceEncoder(device=device)
    encoder_original.eval()
    
    encoder_finetuned = SentenceEncoder(device=device)
    
    tokenizer = BertTokenizer.from_pretrained(LOCKED_BERT_NAME)
    
    model_save_path = FILE_PATHS["bert_finetuned"]
    if os.path.exists(model_save_path):
        state_dict = torch.load(model_save_path, map_location=device)
        _assert_locked_checkpoint(state_dict)
        encoder_finetuned.load_state_dict(state_dict, strict=True)
        encoder_finetuned.eval()
    else:
        encoder_finetuned = encoder_original
    
    if "df_global" not in globals():
        df_articles = pd.read_csv(FILE_PATHS["csv_path"])
    else:
        df_articles = df
    
    matched_dict = {}
    bm25_results_path = FILE_PATHS["bm25_search_results"]
    if os.path.exists(bm25_results_path):
        with open(bm25_results_path, "r", encoding="utf-8") as f:
            matched_dict = json.load(f)
    else:
        pass

    cls_vectors_before = get_all_cls_vectors(df_articles, encoder_original, tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, encoder_finetuned, tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    

    perplexity_before = min(30, max(5, len(cls_vectors_before) // 3))
    perplexity_after = min(30, max(5, len(cls_vectors_after_cpu) // 3))
    
    tsne_before = TSNE(
        n_components=2,
        perplexity=perplexity_before,
        random_state=42,
        max_iter=get_config("tsne_max_iter"),
        verbose=0
    )
    tsne_after = TSNE(
        n_components=2,
        perplexity=perplexity_after,
        random_state=42,
        max_iter=get_config("tsne_max_iter"),
        verbose=0
    )
    projected_2d_before = tsne_before.fit_transform(cls_vectors_before.numpy())
    projected_2d_after = tsne_after.fit_transform(cls_vectors_after_cpu)
    

    group_centers = {}
    if matched_dict:
        for group_name, indices in matched_dict.items():
            if len(indices) > 0:
                valid_indices = [i for i in indices if i < len(projected_2d_after)]
                if len(valid_indices) > 0:
                    group_2d_points = projected_2d_after[valid_indices]
                    group_center_2d = np.mean(group_2d_points, axis=0)
                    group_centers[group_name] = group_center_2d
    
    

    def create_plotly_figure_with_highlights(projected_2d, title, highlighted_indices=None, group_centers=None):
        fig = go.Figure()
        

        article_indices = list(range(len(projected_2d)))
        hover_texts = [f"Article {idx}" for idx in article_indices]
        custom_data = [[idx] for idx in article_indices]
        
        bg_style = PLOT_STYLES["background"]
        fig.add_trace(go.Scatter(
            x=projected_2d[:, 0],
            y=projected_2d[:, 1],
            mode='markers',
            name="Articles",
            marker=dict(
                color=bg_style["color"],
                size=bg_style["size"],
                opacity=bg_style["opacity"],
                symbol="circle",  
                line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
            ),
            customdata=custom_data,
            hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            hovertext=hover_texts
        ))
        

        if highlighted_indices:
            highlighted_x = []
            highlighted_y = []
            highlighted_customdata = []
            highlighted_hovertexts = []
            
            for idx in highlighted_indices:
                if 0 <= idx < len(projected_2d):
                    highlighted_x.append(projected_2d[idx, 0])
                    highlighted_y.append(projected_2d[idx, 1])
                    highlighted_customdata.append([idx])
                    highlighted_hovertexts.append(f"Article {idx}")
            
            if highlighted_x:  
                core_style = PLOT_STYLES["core"]
                fig.add_trace(go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode='markers',
                    name='Highlighted Articles',
                    marker=dict(
                        color=core_style["color"],
                        size=core_style["size"],
                        symbol=core_style["symbol"],
                        line=dict(width=core_style["line_width"], color=core_style["line_color"])
                    ),
                    customdata=highlighted_customdata,
                    hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    hovertext=highlighted_hovertexts
                ))
        

        if "After" in title and group_centers:
            center_style = PLOT_STYLES["center"]
            for group_name, center_2d in group_centers.items():
                color = get_group_color(group_name)  
                fig.add_trace(go.Scatter(
                    x=[center_2d[0]],
                    y=[center_2d[1]],
                    mode='markers+text',
                    name=f'Center: {group_name}',
                    marker=dict(
                        color=color,
                        size=center_style["size"],
                        symbol=center_style["symbol"],
                        opacity=center_style["opacity"],
                        line=dict(width=center_style["line_width"], color=center_style["line_color"])
                    ),
                    text=[group_name],
                    textposition="top center",
                    textfont=dict(size=12, color=color, family='Arial Black'),
                    hovertemplate=f'<b>Group Center: {group_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
        
        layout_style = PLOT_STYLES["layout"]
        fig.update_layout(
            title=title,
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=layout_style["plot_bgcolor"],
            paper_bgcolor=layout_style["paper_bgcolor"],
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(**layout_style["xaxis"], title='TSNE Dimension 1'),
            yaxis=dict(**layout_style["yaxis"], title='TSNE Dimension 2')
        )
        
        return fig
    
    fig_before = create_plotly_figure_with_highlights(projected_2d_before, "Before Training", highlighted_indices, None)
    fig_after = create_plotly_figure_with_highlights(projected_2d_after, "After Training", highlighted_indices, group_centers)
    
    return fig_before, fig_after


_KEYWORD_TSNE_CACHE = None


_ARTICLES_CACHE = {}


_DOCUMENTS_2D_CACHE = {}

def clear_caches():

    global _ARTICLES_CACHE, _DOCUMENTS_2D_CACHE
    _ARTICLES_CACHE.clear()
    _DOCUMENTS_2D_CACHE.clear()
    
    try:
        filtered_path = FILE_PATHS.get("filtered_group_assignment")
        if filtered_path and os.path.exists(filtered_path):
            os.remove(filtered_path)
    except Exception as e:
        pass

    try:
        user_finetuned_path = FILE_PATHS.get("user_finetuned_list")
        if user_finetuned_path and os.path.exists(user_finetuned_path):
            os.remove(user_finetuned_path)

    except Exception as e:
        pass


@app.callback(
    Output('keywords-2d-plot', 'figure'),
    Input('keywords-2d-plot', 'id')  
)
def update_keywords_2d_plot(plot_id):
    global GLOBAL_OUTPUT_DICT, GLOBAL_KEYWORDS, _KEYWORD_TSNE_CACHE
    
    if not GLOBAL_OUTPUT_DICT or not GLOBAL_KEYWORDS:
        if GLOBAL_KEYWORDS:
            pass
        if GLOBAL_OUTPUT_DICT:
            pass
        return {
            'data': [],
            'layout': {
                'title': 'No keywords available',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
    

    try:
        def adjust_text_positions(x_coords, y_coords, keywords, min_distance=0.12):

            import numpy as np
            adjusted_x = x_coords.copy()
            adjusted_y = y_coords.copy()
            

            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            min_dist = min(x_range, y_range) * min_distance
            
            def find_empty_space(current_x, current_y, all_x, all_y, search_radius=0.3):

                search_dist = min(x_range, y_range) * search_radius
                

                for angle in np.linspace(0, 2*np.pi, 16):  
                    for radius in np.linspace(min_dist, search_dist, 8):  
                        test_x = current_x + radius * np.cos(angle)
                        test_y = current_y + radius * np.sin(angle)
                        
                        too_close = False
                        for other_x, other_y in zip(all_x, all_y):
                            if abs(test_x - other_x) < min_dist and abs(test_y - other_y) < min_dist:
                                too_close = True
                                break
                        
                        if not too_close:
                            return test_x, test_y
                
                return current_x + np.random.normal(0, min_dist*0.5), current_y + np.random.normal(0, min_dist*0.5)
            
            for iteration in range(30):
                overlaps_found = 0
                
                for i in range(len(adjusted_x)):
                    has_overlap = False
                    for j in range(len(adjusted_x)):
                        if i != j:
                            if (abs(adjusted_x[i] - adjusted_x[j]) < min_dist and 
                                abs(adjusted_y[i] - adjusted_y[j]) < min_dist):
                                has_overlap = True
                                break
                    
                    if has_overlap:
                        new_x, new_y = find_empty_space(
                            x_coords[i], y_coords[i],  
                            adjusted_x, adjusted_y
                        )
                        adjusted_x[i] = new_x
                        adjusted_y[i] = new_y
                        overlaps_found += 1
                
                if overlaps_found == 0:
                    break
            
            return adjusted_x, adjusted_y
        
        if _KEYWORD_TSNE_CACHE is None:
            keyword_embeddings = embedding_model_kw.encode(GLOBAL_KEYWORDS, convert_to_tensor=True).to(device).cpu().numpy()
            
            perplexity = min(30, max(5, len(keyword_embeddings) // 3))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=0)
            reduced_embeddings = tsne.fit_transform(keyword_embeddings)
            
            x_coords = reduced_embeddings[:, 0]
            y_coords = reduced_embeddings[:, 1]
            x_coords_adjusted, y_coords_adjusted = adjust_text_positions(x_coords, y_coords, GLOBAL_KEYWORDS)
            
            _KEYWORD_TSNE_CACHE = {
                'keywords': GLOBAL_KEYWORDS.copy(),
                'embeddings': reduced_embeddings.copy(),
                'adjusted_x': x_coords_adjusted.copy(),
                'adjusted_y': y_coords_adjusted.copy()
            }
        else:
            reduced_embeddings = _KEYWORD_TSNE_CACHE['embeddings']
            x_coords_adjusted = _KEYWORD_TSNE_CACHE['adjusted_x']
            y_coords_adjusted = _KEYWORD_TSNE_CACHE['adjusted_y']
        
        hover_texts = GLOBAL_KEYWORDS
        
        keyword_colors = ['#2196F3'] * len(GLOBAL_KEYWORDS)
        

        
        fig = {
            'data': [{
                'x': x_coords_adjusted,
                'y': y_coords_adjusted,
                'mode': 'markers+text',  
                'type': 'scatter',
                'marker': {
                    'size': 1,  
                    'color': keyword_colors,
                    'opacity': 0.0  
                },
                'text': GLOBAL_KEYWORDS,  
                'textfont': {
                    'size': 10,  
                    'color': keyword_colors  
                },
                'textposition': 'middle center',
                'hovertext': hover_texts,
                'hoverinfo': 'text',
                'customdata': GLOBAL_KEYWORDS,  
                'hovertemplate': '<b>%{hovertext}</b><extra></extra>',
                'opacity': 1.0  
            }],
            'layout': {
                'title': 'Keywords 2D Visualization',
                'xaxis': {
                    'title': 'Dimension 1',
                    'showgrid': True,
                    'gridcolor': '#f0f0f0'
                },
                'yaxis': {
                    'title': 'Dimension 2', 
                    'showgrid': True,
                    'gridcolor': '#f0f0f0'
                },
                'hovermode': 'closest',
                'clickmode': 'event',  
                'dragmode': 'pan',
                'showlegend': False,
                'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},  
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'size': 10}  
            }
        }
        
        return fig
        
    except Exception as e:
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {
                    'title': 'X',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'yaxis': {
                    'title': 'Y',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12},
                'title': {
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            }
        }


app.clientside_callback(
    """
    function(group_data, current_figure) {
        // Define group colors
        const GROUP_COLORS = {
            "Group 1": "#FF6B6B",
            "Group 2": "#32CD32", 
            "Group 3": "#FF8C00",
            "Group 4": "#8B4513",
            "Group 5": "#FFD700",
            "Group 6": "#8A2BE2",
            "Group 7": "#DC143C",
            "Group 8": "#228B22",
            "Group 9": "#FF1493",
            "Group 10": "#800080",
            "Exclude": "#A9A9A9",
            "Other": "#A9A9A9"
        };
        
        const DEFAULT_COLOR = "#2196F3";
        
        // Check if current figure is valid
        if (!current_figure || !current_figure.data || !current_figure.data[0]) {
            return window.dash_clientside.no_update;
        }
        
        // Clone the figure to avoid mutation
        const new_figure = JSON.parse(JSON.stringify(current_figure));
        const text_data = new_figure.data[0].text;
        
        if (!text_data || !Array.isArray(text_data)) {
            return window.dash_clientside.no_update;
        }
        
        // Update colors based on group assignment
        const new_colors = text_data.map(keyword => {
            if (group_data && typeof group_data === 'object' && group_data[keyword]) {
                const group_name = group_data[keyword];
                // Handle "Exclude" group name
                if (group_name === "Exclude") {
                    return GROUP_COLORS["Exclude"] || DEFAULT_COLOR;
                }
                return GROUP_COLORS[group_name] || DEFAULT_COLOR;
            }
            return DEFAULT_COLOR;
        });
        
        // Update the textfont color and ensure no fading/opacity issues
        new_figure.data[0].textfont.color = new_colors;
        
        // Explicitly set opacity to prevent any fading effects
        new_figure.data[0].opacity = 1.0;
        
        // Ensure marker properties don't cause fading if present
        if (new_figure.data[0].marker) {
            new_figure.data[0].marker.opacity = 1.0;
        }
        
        return new_figure;
    }
    """,
    Output('keywords-2d-plot', 'figure', allow_duplicate=True),
    Input('group-data', 'data'),
    State('keywords-2d-plot', 'figure'),
    prevent_initial_call=True
)





@app.callback(
    Output('documents-2d-plot', 'figure'),
    Input('main-visualization-area', 'children'),  
    [State('display-mode', 'data'),
     State('training-figures', 'data')],  
    prevent_initial_call=True
)
def update_documents_2d_plot_initial(layout_children, display_mode, training_figures):

    global df
    
    
    if display_mode != "keywords":
        raise PreventUpdate
    
    
    if display_mode == "training":
        raise PreventUpdate
    
    if display_mode is None or display_mode not in ["keywords"]:
        raise PreventUpdate
    
    if 'df' not in globals():
        return {
            'data': [],
            'layout': {
                'title': 'No data available',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
    
    try:

        valid_mask = df.iloc[:, 1].notna()
        valid_indices = df.index[valid_mask].tolist()
        all_articles_text = df.loc[valid_indices, df.columns[1]].astype(str).tolist()
        
        assert len(all_articles_text) == len(valid_indices), f"Texts length {len(all_articles_text)} != valid_indices length {len(valid_indices)}"
        truncated_articles = [truncate_text_for_model(text, max_length=256) for text in all_articles_text]
        
        encoder = SentenceEncoder(device=device)
        encoder.eval()
        tokenizer = BertTokenizer.from_pretrained(LOCKED_BERT_NAME)
        
        batch_size = 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(truncated_articles), batch_size):
                batch_texts = truncated_articles[i:i + batch_size]
                
                tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
                if device != 'cpu':
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                
                batch_embeddings = encoder.encode_tokens(tokens).cpu().numpy()
                all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        assert len(document_embeddings) == len(all_articles_text), f"Embeddings length {len(document_embeddings)} != texts length {len(all_articles_text)}"
        
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        perplexity = min(perplexity, len(document_embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1, verbose=0)
        document_2d = tsne.fit_transform(document_embeddings)
        document_2d = document_2d.tolist()
        
        assert len(document_2d) == len(valid_indices), f"document_2d length {len(document_2d)} != valid_indices length {len(valid_indices)}"
        
        
        bg_style = PLOT_STYLES["background"]
        traces = [{
            'x': [document_2d[i][0] for i in range(len(document_2d))],
            'y': [document_2d[i][1] for i in range(len(document_2d))],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'All documents',
            'marker': {
                'size': bg_style["size"],
                'color': bg_style["color"],
                'opacity': bg_style["opacity"],
                'line': {'width': bg_style["line_width"], 'color': bg_style["line_color"]}
            },
            'text': [f'Doc {valid_indices[i]+1}' for i in range(len(document_2d))],
            'customdata': [[valid_indices[i]] for i in range(len(document_2d))],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        }]
        
        layout_style = PLOT_STYLES["layout"]
        fig = {
            'data': traces,
            'layout': {
                'title': {
                    'text': 'Documents 2D Visualization - All Documents',
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'TSNE Dimension 1',
                    **layout_style["xaxis"]
                },
                'yaxis': {
                    'title': 'TSNE Dimension 2',
                    **layout_style["yaxis"]
                },
                'hovermode': 'closest',
                'clickmode': 'event+select',
                'showlegend': True,
                'legend': {
                    'x': 0.02,
                    'y': 0.98,
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',
                    'bordercolor': '#2c3e50',
                    'borderwidth': 1
                },
                'plot_bgcolor': layout_style["plot_bgcolor"],
                'paper_bgcolor': layout_style["paper_bgcolor"],
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12}
            }
        }
        
        return fig
        
    except Exception as e:
        layout_style = PLOT_STYLES["layout"]
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {
                    'title': 'X',
                    **layout_style["xaxis"]
                },
                'yaxis': {
                    'title': 'Y',
                    **layout_style["yaxis"]
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12},
                'title': {
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            }
        }



@app.callback(
    [Output('documents-2d-plot', 'figure', allow_duplicate=True),
     Output('highlighted-indices', 'data', allow_duplicate=True)],
    [Input('selected-keyword', 'data'),
     Input('selected-group', 'data'),  
     Input('selected-article', 'data')],  
    State('group-order', 'data'),  
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_documents_2d_plot(selected_keyword, selected_group, selected_article, group_order, display_mode):

    global df, _DOCUMENTS_2D_CACHE
    
    if display_mode != "keywords":

        raise PreventUpdate
    
    
    if display_mode == "training":

        raise PreventUpdate
    
    
    if display_mode is None or display_mode not in ["keywords"]:
        raise PreventUpdate
    
    
    
    if 'df' not in globals():
        return {
            'data': [],
            'layout': {
                'title': 'No data available',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }, []
    
    base_cache_key = "docs_base_figure"
    
    if base_cache_key in _DOCUMENTS_2D_CACHE:
        base_fig = _DOCUMENTS_2D_CACHE[base_cache_key]
        base_traces = base_fig.get('data', [])
        base_layout = base_fig.get('layout', {})
        
        if len(base_traces) > 0:
            traces = [base_traces[0].copy()]
            
            keyword_group_indices = []
            selected_article_indices = []
            snapshot_before = get_latest_training_snapshot("before")
            
            if selected_keyword:
                _, valid_idx_to_doc2d_idx = get_valid_doc2d_index_map(df)
                if snapshot_before and snapshot_before.get("keyword_matches_in_group"):
                    keyword_group = resolve_group_for_keyword(group_order, selected_keyword)
                    if keyword_group:
                        matched_docs = snapshot_before["keyword_matches_in_group"].get(keyword_group, {}).get(selected_keyword, [])
                        keyword_group_indices = [valid_idx_to_doc2d_idx[i] for i in matched_docs if i in valid_idx_to_doc2d_idx]
                    else:
                        keyword_group_indices = []
                else:
                    matched_docs = get_keyword_doc_indices_cached(selected_keyword, df)
                    keyword_group_indices = [valid_idx_to_doc2d_idx[i] for i in matched_docs if i in valid_idx_to_doc2d_idx]
            
            elif selected_group and group_order:
                group_keywords = []
                for group_name, keywords in group_order.items():
                    if group_name == selected_group:
                        group_keywords = keywords
                        break
                
                _, valid_idx_to_doc2d_idx = get_valid_doc2d_index_map(df)
                if snapshot_before and snapshot_before.get("group_docs"):
                    matched_docs = snapshot_before["group_docs"].get(selected_group, [])
                    keyword_group_indices = [valid_idx_to_doc2d_idx[i] for i in matched_docs if i in valid_idx_to_doc2d_idx]
                else:
                    matched_docs = get_group_doc_indices_cached(group_keywords, df)
                    keyword_group_indices = [valid_idx_to_doc2d_idx[i] for i in matched_docs if i in valid_idx_to_doc2d_idx]
            
            if selected_article is not None:
                _, valid_idx_to_doc2d_idx = get_valid_doc2d_index_map(df)
                if selected_article in valid_idx_to_doc2d_idx:
                    doc2d_idx = valid_idx_to_doc2d_idx[selected_article]
                    selected_article_indices = [doc2d_idx]
            
            if len(keyword_group_indices) > 0:
                core_style = PLOT_STYLES["core"]
                traces.append({
                    'x': [base_traces[0]['x'][i] for i in keyword_group_indices],
                    'y': [base_traces[0]['y'][i] for i in keyword_group_indices],
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': 'Keyword/Group matches',
                    'marker': {
                        'size': core_style["size"],
                        'color': core_style["color"],
                        'symbol': core_style["symbol"],
                        'line': {'width': core_style["line_width"], 'color': core_style["line_color"]}
                    },
                    'text': [base_traces[0]['text'][i] for i in keyword_group_indices],
                    'customdata': [base_traces[0]['customdata'][i] for i in keyword_group_indices],
                    'hovertemplate': '<b>%{text}</b><extra></extra>'
                })
            
            if len(selected_article_indices) > 0:
                traces.append({
                    'x': [base_traces[0]['x'][i] for i in selected_article_indices],
                    'y': [base_traces[0]['y'][i] for i in selected_article_indices],
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': 'Selected Article',
                    'marker': {
                        'size': 20,
                        'color': '#FF0000',
                        'symbol': 'star',
                        'line': {'width': 3, 'color': 'white'}
                    },
                    'text': [base_traces[0]['text'][i] for i in selected_article_indices],
                    'customdata': [base_traces[0]['customdata'][i] for i in selected_article_indices],
                    'hovertemplate': '<b>%{text}</b><extra></extra>'
                })
            
            highlighted_indices = list(keyword_group_indices) + list(selected_article_indices)
            
            title_parts = []
            if selected_keyword:
                title_parts.append(f"Keyword: '{selected_keyword}'")
            elif selected_group:
                title_parts.append(f"Group: '{selected_group}'")
            if selected_article is not None:
                title_parts.append(f"Selected Article {selected_article + 1}")
            
            if title_parts:
                title = f"Documents 2D Visualization - {' | '.join(title_parts)}"
            else:
                title = "Documents 2D Visualization"
            
            fig = {
                'data': traces,
                'layout': {
                    **base_layout,
                    'title': {
                        'text': title,
                        'font': {'size': 16, 'color': '#2c3e50'},
                        'x': 0.5,
                        'xanchor': 'center'
                    }
                }
            }
            
            return fig, highlighted_indices
    
    try:
        
        if _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
            document_embeddings = _GLOBAL_DOCUMENT_EMBEDDINGS
            tsne_result = get_document_tsne()
            if tsne_result is None:
                raise ValueError("Failed to compute document t-SNE")
            document_2d = tsne_result.tolist()
            
        else:
            pass
        valid_mask = df.iloc[:, 1].notna()
        valid_indices = df.index[valid_mask].tolist()
        all_articles_text = df.loc[valid_indices, df.columns[1]].astype(str).tolist()
        
        truncated_articles = [truncate_text_for_model(text, max_length=256) for text in all_articles_text]
        
        encoder = SentenceEncoder(device=device)
        encoder.eval()
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
        
        document_embeddings = np.array(all_embeddings)
        
        assert len(document_embeddings) == len(all_articles_text), f"Embeddings length {len(document_embeddings)} != texts length {len(all_articles_text)}"
        
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        perplexity = min(perplexity, len(document_embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1, verbose=0)
        document_2d = tsne.fit_transform(document_embeddings)
        document_2d = document_2d.tolist()
        
        assert len(document_2d) == len(valid_indices), f"document_2d length {len(document_2d)} != valid_indices length {len(valid_indices)}"
        
        
        valid_idx_to_doc2d_idx = {valid_idx: i for i, valid_idx in enumerate(valid_indices)}
        
        highlight_mask = []
        highlight_reason = ""
        
        if selected_keyword:
            for valid_idx in valid_indices:
                text = str(df.loc[valid_idx, df.columns[1]])
                contains_keyword = contains_keyword_word_boundary(text, selected_keyword)
                highlight_mask.append(contains_keyword)
                if contains_keyword:
                    pass
            highlight_reason = f"Documents containing '{selected_keyword}'"
        
        elif selected_group and group_order:
            group_keywords = []
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    group_keywords = keywords
                    break
            
            
            for valid_idx in valid_indices:
                text = str(df.loc[valid_idx, df.columns[1]])
                contains_group_keyword = any(contains_keyword_word_boundary(text, keyword) for keyword in group_keywords)
                highlight_mask.append(contains_group_keyword)
            
            highlight_reason = f"Documents containing keywords from group '{selected_group}'"
        
        else:
            highlight_mask = [False] * len(valid_indices)
            highlight_reason = ""
        
        selected_article_mask = [False] * len(valid_indices)
        if selected_article is not None and selected_article in valid_idx_to_doc2d_idx:
            doc2d_idx = valid_idx_to_doc2d_idx[selected_article]
            selected_article_mask[doc2d_idx] = True
        
        bg_style = PLOT_STYLES["background"]
        traces = []
        
        bg_trace = {
            'x': [document_2d[i][0] for i in range(len(document_2d))],
            'y': [document_2d[i][1] for i in range(len(document_2d))],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'All documents',
            'marker': {
                'size': bg_style["size"],
                'color': bg_style["color"],
                'opacity': bg_style["opacity"],
                'line': {'width': bg_style["line_width"], 'color': bg_style["line_color"]}
            },
            'text': [f'Doc {valid_indices[i]+1}' for i in range(len(document_2d))],
            'customdata': [[valid_indices[i]] for i in range(len(document_2d))],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        }
        traces.append(bg_trace)
        
        keyword_group_indices = np.where(np.array(highlight_mask))[0]
        selected_article_indices = np.where(np.array(selected_article_mask))[0]
        
        if len(keyword_group_indices) > 0:
            core_style = PLOT_STYLES["core"]
            traces.append({
                'x': [document_2d[i][0] for i in keyword_group_indices],
                'y': [document_2d[i][1] for i in keyword_group_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Keyword/Group matches',
                'marker': {
                    'size': core_style["size"],
                    'color': core_style["color"],
                    'symbol': core_style["symbol"],
                    'line': {'width': core_style["line_width"], 'color': core_style["line_color"]}
                },
                'text': [f'Doc {valid_indices[i]+1}' for i in keyword_group_indices],
                'customdata': [[valid_indices[i]] for i in keyword_group_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        
        if len(selected_article_indices) > 0:
            traces.append({
                'x': [document_2d[i][0] for i in selected_article_indices],
                'y': [document_2d[i][1] for i in selected_article_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Selected Article',
                'marker': {
                    'size': 20,
                    'color': '#FF0000',  
                    'symbol': 'star',
                    'line': {'width': 3, 'color': 'white'}
                },
                'text': [f'Doc {valid_indices[i]+1}' for i in selected_article_indices],
                'customdata': [[valid_indices[i]] for i in selected_article_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        
        title_parts = []
        
        if selected_keyword:
            title_parts.append(f"Keyword: '{selected_keyword}'")
        elif selected_group and highlight_reason:
            title_parts.append(highlight_reason)
        
        if selected_article is not None:
            title_parts.append(f"Selected Article {selected_article + 1}")
        
        if title_parts:
            title = f"Documents 2D Visualization - {' | '.join(title_parts)}"
        else:
            title = "Documents 2D Visualization"
        
        fig = {
            'data': traces,
            'layout': {
                'title': {
                    'text': title,
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'TSNE Dimension 1',
                    **PLOT_STYLES["layout"]["xaxis"]
                },
                'yaxis': {
                    'title': 'TSNE Dimension 2',
                    **PLOT_STYLES["layout"]["yaxis"]
                },
                'hovermode': 'closest',
                'showlegend': True,
                'legend': {
                    'x': 0.02,
                    'y': 0.98,
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',
                    'bordercolor': '#2c3e50',
                    'borderwidth': 1
                },
                'plot_bgcolor': PLOT_STYLES["layout"]["plot_bgcolor"],
                'paper_bgcolor': PLOT_STYLES["layout"]["paper_bgcolor"],
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12}
            }
        }
        
        highlighted_indices = []
        if len(keyword_group_indices) > 0:
            highlighted_indices.extend(keyword_group_indices.tolist())
        if len(selected_article_indices) > 0:
            highlighted_indices.extend(selected_article_indices.tolist())
        
        
        base_cache_key = "docs_base_figure"
        if base_cache_key not in _DOCUMENTS_2D_CACHE:
            base_fig = {
                'data': [traces[0].copy()],
                'layout': fig['layout'].copy()
            }
            _DOCUMENTS_2D_CACHE[base_cache_key] = base_fig
        
        return fig, highlighted_indices
        
    except Exception as e:
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {
                    'title': 'X',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'yaxis': {
                    'title': 'Y',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12},
                'title': {
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            }
        }, []

@app.callback(
    [Output("group-data", "data", allow_duplicate=True),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input("keywords-2d-plot", "clickData"),
    [State("selected-group", "data"),
     State("group-data", "data"),
     State("display-mode", "data")],
    prevent_initial_call=True
)
def handle_plot_click(click_data, selected_group, group_data, display_mode):
    
    if not click_data:
        raise PreventUpdate
    
    try:
        clicked_keyword = click_data['points'][0]['customdata']
        
        if selected_group:
            new_data = dict(group_data) if group_data else {}
            if clicked_keyword in new_data and new_data[clicked_keyword]:
                if new_data[clicked_keyword] != selected_group:
                    pass
                else:
                    pass
            else:
                pass
            new_data[clicked_keyword] = selected_group
            
            if display_mode == "training":
                return new_data, dash.no_update
            else:
                return new_data, clicked_keyword  
        else:
            
            if display_mode == "training":
                return group_data, dash.no_update
            else:
                return group_data, clicked_keyword
        
    except Exception as e:
        raise PreventUpdate

@app.callback(
    [Output("train-btn", "children"),
     Output("train-btn", "style"),
     Output("train-btn", "disabled"),
     Output("switch-view-btn", "style"),
     Output("display-mode", "data"),
     Output("training-figures", "data"),
     Output("gap-filter-warning", "children")],
    Input("train-btn", "n_clicks"),
    State("group-order", "data"),
    prevent_initial_call=True
)
def handle_train_button(n_clicks, group_order):
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    normal_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#4CAF50",
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "pointer"
    }
    
    training_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#FF9800",  
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "not-allowed",
        "animation": "pulse 1.5s infinite"
    }
    
    if not group_order:
        empty_fig = {
            'data': [],
            'layout': {
                'title': 'No group data available for training',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
        return "Train", normal_style, False, {"display": "none"}, "keywords", {"before": empty_fig, "after": empty_fig}, ""
    
    try:
        
        training_group_order = group_order
        
        with open(FILE_PATHS["final_list_path"], "w", encoding="utf-8") as f:
            json.dump(training_group_order, f, indent=4, ensure_ascii=False)
        
        if os.path.exists(FILE_PATHS["final_list_path"]):
            with open(FILE_PATHS["final_list_path"], "r", encoding="utf-8") as f:
                saved_data = json.load(f)
        else:
            raise FileNotFoundError(f"Could not save group data to {FILE_PATHS['final_list_path']}")
        
        # 记录训练前的状态
        record_user_data("training_before", group_order=training_group_order)
        
        training_fig = {
            'data': [],
            'layout': {
                'title': 'Training in Progress...',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'},
                'annotations': [{
                    'text': 'Training model in progress...<br>This may take several minutes',
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 18, 'color': '#FF9800'}
                }]
            }
        }
        
        try:
            fig_before, fig_after, gap_warning_text = run_training()
        except Exception as e:
            traceback.print_exc()
            
            error_fig = {
                'data': [],
                'layout': {
                    'title': f'Training Failed: {str(e)}',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': f'Training failed with error:<br>{str(e)}<br><br>Check console for details.',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#f44336'}
                    }]
                }
            }
            return "Train (Failed)", normal_style, False, {"display": "block"}, "keywords", {"before": error_fig, "after": error_fig}, ""
        
        if fig_before is None or fig_after is None:
            error_fig = {
                'data': [],
                'layout': {
                    'title': 'Training Failed',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': f'Training failed.<br>{gap_warning_text or "Check console for details."}',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#f44336'}
                    }]
                }
            }
            return "Train (Failed)", normal_style, False, {"display": "block"}, "keywords", {"before": error_fig, "after": error_fig}, gap_warning_text or ""
        
        group_info_path = FILE_PATHS["training_group_info"]
        with open(group_info_path, "w", encoding="utf-8") as f:
            json.dump(group_order, f, indent=4, ensure_ascii=False)
        
        if hasattr(fig_before, 'data') and fig_before.data:
            if len(fig_before.data) > 0:
                first_trace = fig_before.data[0]
        
        if hasattr(fig_after, 'data') and fig_after.data:
            if len(fig_after.data) > 0:
                first_trace = fig_after.data[0]
        
        completed_style = {
            "margin-top": "20px",
            "padding": "10px 20px",
            "fontSize": "16px",
            "backgroundColor": "#2E7D32",  
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer"
        }
        
        switch_button_style = {
            "margin": "15px auto",
            "padding": "12px 30px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "backgroundColor": "#3498db",
            "color": "white",
            "border": "none",
            "borderRadius": "6px",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 10px rgba(52, 152, 219, 0.3)",
            "display": "block"  
        }
        
        
        import numpy as np
        
        def fig_to_serializable_dict(fig):
            result = {
            'data': [],
                'layout': {}
            }
            
            if hasattr(fig, 'layout'):
                result['layout'] = fig.layout.to_plotly_json() if hasattr(fig.layout, 'to_plotly_json') else {}
            
            for trace in fig.data:
                trace_dict = {}
                
                for attr in ['x', 'y', 'mode', 'type', 'name', 'text', 'textposition', 'textfont', 'customdata', 'hovertemplate', 'hovertext']:
                    if hasattr(trace, attr):
                        val = getattr(trace, attr)
                        if val is not None:
                 
                            if hasattr(val, 'tolist'):
                                trace_dict[attr] = val.tolist()
                            elif hasattr(val, '__iter__') and not isinstance(val, str):
                                trace_dict[attr] = list(val)
                            else:
                                trace_dict[attr] = val
                
                if hasattr(trace, 'marker'):
                    marker_dict = {}
                    marker = trace.marker
                    for m_attr in ['color', 'size', 'symbol', 'opacity', 'line']:
                        if hasattr(marker, m_attr):
                            m_val = getattr(marker, m_attr)
                            if m_val is not None:
                                if m_attr == 'line' and hasattr(m_val, 'to_plotly_json'):
                                    marker_dict[m_attr] = m_val.to_plotly_json()
                                elif hasattr(m_val, 'tolist'):
                                    marker_dict[m_attr] = m_val.tolist()
                                else:
                                    marker_dict[m_attr] = m_val
                    trace_dict['marker'] = marker_dict
                
                trace_name = trace_dict.get('name', 'Unknown')
                x_len = len(trace_dict.get('x', []))
                
                result['data'].append(trace_dict)
            
            return result
        
        fig_before_dict = fig_to_serializable_dict(fig_before)
        fig_after_dict = fig_to_serializable_dict(fig_after)
        
        if fig_after_dict.get('data'):
            for trace in fig_after_dict['data']:
                trace_name = trace.get('name', 'Unknown')
                x_len = len(trace.get('x', []))
                if 'Center:' in trace_name:
                    pass

        if isinstance(fig_before_dict, dict) and 'data' in fig_before_dict:
            if len(fig_before_dict['data']) > 0:
                first_trace = fig_before_dict['data'][0]
        
        if isinstance(fig_after_dict, dict) and 'data' in fig_after_dict:
            if len(fig_after_dict['data']) > 0:
                first_trace = fig_after_dict['data'][0]
        
        return "Training Complete", completed_style, False, switch_button_style, "training", {"before": fig_before_dict, "after": fig_after_dict}, gap_warning_text
        
    except Exception as e:
           
        traceback.print_exc()
        
        error_fig = {
            'data': [],
            'layout': {
                'title': f'Training Error: {str(e)}',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'},
                'annotations': [{
                    'text': f'Training failed: {str(e)}',
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'red'}
                }]
            }
        }
        
        error_style = {
            "margin-top": "20px",
            "padding": "10px 20px",
            "fontSize": "16px",
            "backgroundColor": "#F44336",  
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer"
        }
        
        switch_button_style = {"display": "none"}
        
        return "Training Failed", error_style, False, switch_button_style, "keywords", {"before": None, "after": None}, ""

@app.callback(
    [Output("train-btn", "children", allow_duplicate=True),
     Output("train-btn", "style", allow_duplicate=True),
     Output("train-btn", "disabled", allow_duplicate=True)],
    Input("train-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_train_button_immediately(n_clicks):

    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    training_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#FF9800",  
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "not-allowed",
        "opacity": "0.8",
        "transform": "scale(0.98)"
    }
    
    return "Training...", training_style, True





@app.callback(
    [Output("article-fulltext-container", "children", allow_duplicate=True),
     Output("highlighted-indices", "data", allow_duplicate=True)],
    [Input("plot-before", "clickData"),
     Input("plot-after", "clickData")],
    prevent_initial_call=True
)
def display_article_content_training(click_data_before, click_data_after):

    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    click_data = None
    if ctx.triggered[0]['prop_id'] == 'plot-before.clickData':
        click_data = click_data_before
    elif ctx.triggered[0]['prop_id'] == 'plot-after.clickData':
        click_data = click_data_after
    
    if not click_data:
        raise PreventUpdate
    
    try:
        article_index = click_data['points'][0]['customdata'][0]
        
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        if article_index is not None and article_index < len(df):
            article_text = str(df.iloc[article_index, 1])
            
            content = html.Div([
                html.H5(f"Article {article_index + 1}", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
            
            return content, [article_index]
        else:
            return html.P("Article not found", style={"color": "red"}), []
    
    except Exception as e:
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), []


@app.callback(
    [Output("display-mode", "data", allow_duplicate=True),
     Output("switch-view-btn", "children")],
    Input("switch-view-btn", "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def switch_display_mode(n_clicks, current_mode):

    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    
    if current_mode == "keywords":
        new_mode = "training"
        button_text = "Switch to Keywords View"
    elif current_mode == "training":
        new_mode = "keywords"
        button_text = "Switch to Training View"
    elif current_mode == "finetune":
        new_mode = "training"
        button_text = "Switch to Training View"
    else:
        new_mode = "keywords"
        button_text = "Switch to Training View"
    
    return new_mode, button_text

@app.callback(
    [Output("switch-view-btn", "style", allow_duplicate=True),
     Output("switch-view-btn", "children", allow_duplicate=True)],
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_switch_view_btn_visibility(display_mode):
    base_style = {
        "backgroundColor": "#3498db",
        "color": "white",
        "border": "none",
        "padding": "10px 20px",
        "borderRadius": "6px",
        "fontSize": "1rem",
        "fontWeight": "bold",
        "cursor": "pointer",
        "transition": "all 0.3s ease",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
        "marginRight": "10px",
        "minWidth": "180px",
        "flexShrink": "0"
    }
    
    if display_mode == "finetune":
        base_style["display"] = "block"
        button_text = "Switch to Training View"  
    elif display_mode == "training":
        base_style["display"] = "block"
        button_text = "Switch to Keywords View"  
    elif display_mode == "keywords":
        base_style["display"] = "none"
        button_text = "Switch to Training View"  
    else:
        base_style["display"] = "block"
        button_text = "Switch to Training View"  
    
    return base_style, button_text

@app.callback(
    Output("switch-finetune-btn", "style"),
    [Input("display-mode", "data"), Input("training-figures", "data")]
)
def show_switch_finetune_btn(display_mode, training_figures):
    
    base_style = {
        "margin": "15px auto",
        "padding": "12px 30px",
        "fontSize": "1rem",
        "fontWeight": "bold",
        "backgroundColor": "#8e44ad",
        "color": "white",
        "border": "none",
        "borderRadius": "6px",
        "cursor": "pointer",
        "transition": "all 0.3s ease",
        "boxShadow": "0 3px 10px rgba(142, 68, 173, 0.3)",
        "display": "none"
    }
    try:
        has_after = isinstance(training_figures, dict) and bool(training_figures.get("after"))
        
        if display_mode in ("training", "finetune", "keywords") and has_after:
            base_style["display"] = "block"
        else:
            base_style["display"] = "none"
    except Exception as e:
        base_style["display"] = "none"
    
    return base_style
            

@app.callback(
    [Output("main-visualization-area", "children"),
     Output("training-group-management-area", "style"),
     Output("keywords-group-management-area", "style"),
     Output("finetune-group-management-area", "style")],
    [Input("display-mode", "data"),
     Input("training-figures", "data")],
    prevent_initial_call=True
)
def update_main_visualization_area(display_mode, training_figures):
   
    if training_figures:
        pass

    if display_mode == "training":
       
        if training_figures:
            fig_before = training_figures.get("before", {})
            fig_after = training_figures.get("after", {})
        else:
            fig_before = {
                'data': [],
                'layout': {
                    'title': 'Before Training - No Data Available',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': 'Please run training first to see results',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#666'}
                    }]
                }
            }
            fig_after = {
                'data': [],
                'layout': {
                    'title': 'After Training - No Data Available',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': 'Please run training first to see results',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#666'}
                    }]
                }
            }
        
        training_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        
        return [
            html.Div([
                html.H4("Before Training", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Training results before model optimization", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='plot-before',
                    figure=fig_before if fig_before else {},
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '1%'
            }),
            
            html.Div([
                html.H4("After Training", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Training results after model optimization", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='plot-after',
                    figure=fig_after if fig_after else {},
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '1%'
            })
        ], training_group_style, {'display': 'none', 'marginBottom': '30px'}, {'display': 'none', 'marginBottom': '30px'}
    elif display_mode == "finetune":
        
        if training_figures:
            fig_after = training_figures.get("after", {})
        else:
            fig_after = {
                'data': [],
                'layout': {
                    'title': 'Finetune Mode - No Training Data Available',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': 'Please run training first to access finetune mode',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#666'}
                    }]
                }
            }
        
        finetune_group_style = {'display': 'flex', 'marginBottom': '30px'}

        
        return [
            html.Div([
                html.H4("Finetune Mode - Interactive 2D", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Click on points to preview text and reassign samples", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='finetune-2d-plot',
                    figure=fig_after if fig_after else {},
                    style={'height': '800px'},  
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '100%',
                'minHeight': '850px',  
                'padding': '20px',
                'margin': '0 auto',
                'display': 'block'
            })
        ], {'display': 'none', 'marginBottom': '30px'}, {'display': 'none', 'marginBottom': '30px'}, finetune_group_style
    else:

        
        training_group_style = {'display': 'none', 'marginBottom': '30px'}
        keywords_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        return [
            html.Div([
                html.H4("Keywords 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Click on keywords to highlight related documents", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='keywords-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '1%'
            }),
            
            html.Div([
                html.H4("Documents 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Documents highlighted by selected keyword", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='documents-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '1%'
            })
        ], training_group_style, keywords_group_style, {'display': 'none', 'marginBottom': '30px'}


@app.callback(
    Output('highlighted-indices', 'data', allow_duplicate=True),
    [Input('training-selected-keyword', 'data'),
     Input('training-selected-group', 'data')],
    State('group-order', 'data'),
    State('training-figures', 'data'),
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_training_highlights(selected_keyword, selected_group, group_order, training_figures, display_mode):
    global df
    

    if display_mode != "training":
        raise PreventUpdate
    
    
    if 'df' not in globals() or not training_figures:
        return {"type": "none", "indices": []}
    
    snapshot_after = get_latest_training_snapshot("after")
    
    
    if selected_keyword:

        keyword_indices = []
        
        if snapshot_after and snapshot_after.get("keyword_matches_in_group"):
            keyword_group = resolve_group_for_keyword(group_order, selected_keyword)
            if keyword_group:
                keyword_indices = snapshot_after["keyword_matches_in_group"].get(keyword_group, {}).get(selected_keyword, [])
                return {"type": "keyword", "indices": keyword_indices, "keyword": selected_keyword}
        

        filtered_path = FILE_PATHS["filtered_group_assignment"]
        if os.path.exists(filtered_path):
            try:
                with open(filtered_path, "r", encoding="utf-8") as f:
                    filtered_dict = json.load(f)
                

                keyword_group = None
                for grp_name, keywords in group_order.items():
                    if selected_keyword in keywords:
                        keyword_group = grp_name
                        break
                
                if keyword_group and keyword_group in filtered_dict:
   
                    group_filtered_docs = filtered_dict[keyword_group]
                    

                    for idx in group_filtered_docs:
                        if idx < len(df):
                            text = str(df.iloc[idx, 1])
                            if contains_keyword_word_boundary(text, selected_keyword):
                                keyword_indices.append(idx)
                    
                    return {"type": "keyword", "indices": keyword_indices, "keyword": selected_keyword}
                else:
                    return {"type": "keyword", "indices": [], "keyword": selected_keyword}
            except Exception as e:
                return {"type": "keyword", "indices": [], "keyword": selected_keyword}
        else:
            return {"type": "keyword", "indices": [], "keyword": selected_keyword}
        
    elif selected_group and group_order:
        
        if selected_group in group_order:
            group_keywords = group_order[selected_group]
            
            group_indices = []
            if snapshot_after and snapshot_after.get("group_docs"):
                group_indices = snapshot_after["group_docs"].get(selected_group, [])
                return {"type": "group", "indices": group_indices, "group": selected_group}
            
            filtered_path = FILE_PATHS["filtered_group_assignment"]
            if os.path.exists(filtered_path):
                try:
                    with open(filtered_path, "r", encoding="utf-8") as f:
                        filtered_dict = json.load(f)
                    
                    if selected_group in filtered_dict:
                        group_indices = filtered_dict[selected_group]
                        return {"type": "group", "indices": group_indices, "group": selected_group}
                    else:
                        pass
                except Exception as e:
                    pass

            try:
                from rank_bm25 import BM25Okapi
                from nltk.stem import SnowballStemmer
                import re
                
                stemmer = SnowballStemmer('english')
                
                def process_articles_serial(articles):      
                    tokenized_corpus, valid_indices = [], []
                    for i, a in enumerate(articles):
                        try:
                            a = re.sub(r'\d+', '', str(a)).strip()
                            if len(a.split()) >= 5:
                                words = word_tokenize(a)
                                stemmed = [stemmer.stem(w.lower()) for w in words]
                                if stemmed:
                                    tokenized_corpus.append(" ".join(stemmed))
                                    valid_indices.append(i)
                        except Exception:
                            pass
                    return tokenized_corpus, valid_indices
                
                all_texts = [str(df.iloc[i, 1]) for i in range(len(df))]
                tokenized_corpus, valid_indices = process_articles_serial(all_texts)
                bm25 = BM25Okapi([s.split() for s in tokenized_corpus])
                
                query_tokens = []
                for kw in group_keywords:
                    q = [stemmer.stem(w.lower()) for w in word_tokenize(kw)]
                    query_tokens.extend(q)
                
                scores = bm25.get_scores(query_tokens)
                
                idx_corpus = [i for i, s in enumerate(scores) if s > 0.1]
                if len(idx_corpus) == 0:
                    idx_corpus = [i for i, s in enumerate(scores) if s > 0.01]
                
                idx_orig = [valid_indices[i] for i in idx_corpus]
                
                score_idx_pairs = [(scores[i], valid_indices[i]) for i in idx_corpus]
                score_idx_pairs.sort(reverse=True)
                
                group_indices = [idx for _, idx in score_idx_pairs[:100]]


                
            except Exception as e:
                for i, text in enumerate(df.iloc[:, 1]):
                    text_str = str(text)
                    if any(contains_keyword_word_boundary(text_str, keyword) for keyword in group_keywords):
                        group_indices.append(i)
            
            return {"type": "group", "indices": group_indices, "group": selected_group}
        else:
            return {"type": "group", "indices": [], "group": selected_group}
    
    return {"type": "none", "indices": []}

@app.callback(
    [Output('plot-before', 'figure', allow_duplicate=True),
     Output('plot-after', 'figure', allow_duplicate=True)],
    [Input('highlighted-indices', 'data'),
     Input('training-selected-article', 'data'),
     Input('display-mode', 'data')],
    [State('training-figures', 'data'),
     State('group-order', 'data')],
    prevent_initial_call=True
)
def update_training_plots_with_highlights(highlighted_indices, training_selected_article, display_mode, training_figures, group_order):

    global df
    

    

    if display_mode != "training":
        raise PreventUpdate
    
    

    if not training_figures:
        return {}, {}
    
    

    fig_before = training_figures.get("before", {})
    fig_after = training_figures.get("after", {})
    

    keyword_group_highlights = []
    selected_article_highlight = None
    

    if isinstance(highlighted_indices, dict) and 'type' in highlighted_indices:
        highlight_type = highlighted_indices.get('type')
        highlight_indices = highlighted_indices.get('indices', [])
        
        
        if highlight_type == "group":

            keyword_group_highlights = highlight_indices
            
        elif highlight_type == "keyword":

            keyword_group_highlights = highlight_indices
            
        elif highlight_type == "none":

            keyword_group_highlights = []
    

    if training_selected_article is not None and training_selected_article < len(df):
        selected_article_highlight = training_selected_article
        
        if keyword_group_highlights and training_selected_article not in keyword_group_highlights:
            pass
        elif keyword_group_highlights and training_selected_article in keyword_group_highlights:
            pass
        else:
            pass

    updated_fig_before = apply_highlights_to_training_plot(fig_before, keyword_group_highlights, selected_article_highlight, "before")
    updated_fig_after = apply_highlights_to_training_plot(fig_after, keyword_group_highlights, selected_article_highlight, "after")
    
    return updated_fig_before, updated_fig_after

def apply_highlights_to_training_plot(fig, keyword_group_highlights, selected_article_highlight, plot_name):

    if not fig or 'data' not in fig:
        return fig
    
    
    updated_fig = fig.copy()
    
    if not updated_fig['data']:
        return updated_fig
    
    traces = []
    main_trace = None
    center_traces = []
    
    
    for i, trace in enumerate(updated_fig['data']):
        trace_name = trace.get('name', 'Unknown')
        marker = trace.get('marker', {})
        symbol = marker.get('symbol', 'circle')
        x_len = len(trace.get('x', []))
        
        
        if symbol == 'diamond' or 'Center' in trace_name:
            center_traces.append(trace)
        elif main_trace is None and x_len > 10 and symbol != 'star':
            main_trace = trace
    
    
    if main_trace:
        traces.append(main_trace)
    else:
        return fig
    
    traces.extend(center_traces)
    
    x_data = main_trace['x'] if isinstance(main_trace['x'], (list, tuple)) else list(main_trace['x'])
    y_data = main_trace['y'] if isinstance(main_trace['y'], (list, tuple)) else list(main_trace['y'])
        
    if keyword_group_highlights:
        highlight_x = [x_data[i] for i in keyword_group_highlights if i < len(x_data)]
        highlight_y = [y_data[i] for i in keyword_group_highlights if i < len(y_data)]
        
        if highlight_x and highlight_y:
            traces.append({
                'x': highlight_x,
                'y': highlight_y,
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Selected Group',
                'marker': {
                    'size': 15,
                    'color': '#FFD700',  
                    'symbol': 'star',
                    'line': {'width': 2, 'color': 'white'}
                },
                'text': [f'Doc {i+1}' for i in keyword_group_highlights if i < len(x_data)],
                'customdata': [[i] for i in keyword_group_highlights if i < len(x_data)],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
    
    if selected_article_highlight is not None and selected_article_highlight < len(x_data):
        article_x = [x_data[selected_article_highlight]]
        article_y = [y_data[selected_article_highlight]]
        
        traces.append({
            'x': article_x,
            'y': article_y,
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Selected Article',
            'marker': {
                'size': 20,
                'color': '#FF0000',  
                'symbol': 'star',
                'line': {'width': 3, 'color': 'white'}
            },
            'text': [f'Doc {selected_article_highlight+1}'],
            'customdata': [[selected_article_highlight]],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        })
    
    updated_fig['data'] = traces
    
    
    return updated_fig

@app.callback(
    Output("training-group-containers", "children"),
    [Input("group-order", "data"),
     Input("training-selected-group", "data"),
     Input("display-mode", "data")],  
    [State("training-selected-keyword", "data")],
    prevent_initial_call=False  
)
def render_training_groups(group_order, selected_group, display_mode, selected_keyword):

    
    if display_mode != "training":
        raise PreventUpdate
    
    
    if not group_order:
        return html.Div([
            html.H6("Training Group Management", style={"color": "#2c3e50", "marginBottom": "10px"}),
            html.P("No groups have been created yet. Please create groups in Keywords mode first.", 
                   style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
        ])
    

    children = []
    for grp_name, kw_list in group_order.items():
        if grp_name == "Exclude":

            if kw_list:  
                group_display_name = "Exclude"
                group_color = get_group_color(grp_name)
            else:  
                group_display_name = "Other (Exclude)"
                group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Training Group {group_number}"
            group_color = get_group_color(grp_name)
        
        if grp_name == "Exclude":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  
            }
        else:
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px solid {group_color}",
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px"
            }
        
        group_header = html.Button(
            group_display_name,
            id={"type": "training-group-header", "index": grp_name},
            style=header_style
        )

        group_keywords = []
        for i, kw in enumerate(kw_list):

            is_selected = selected_keyword and kw == selected_keyword
            
            keyword_button = html.Button(
                kw,
                id={"type": "training-select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  
                    "color": "white" if is_selected else group_color,  
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  
                }
            )
            
            keyword_item = html.Div([
                keyword_button,
                html.Button("×", id={"type": "training-remove-keyword", "group": grp_name, "index": i}, 
                           style={"margin": "2px", "padding": "2px 6px", "fontSize": "10px", "color": "red", "float": "right"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"})
            group_keywords.append(keyword_item)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "8px",
                "minHeight": "50px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "10px"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"marginBottom": "15px"}
        )
        children.append(group_container)

    return children

@app.callback(
    [Output("training-selected-group", "data"),
     Output("training-selected-keyword", "data")],
    [Input({"type": "training-group-header", "index": ALL}, "n_clicks"),
     Input("display-mode", "data")],
    prevent_initial_call=True
)
def select_training_group(n_clicks, display_mode):

    ctx = dash.callback_context
    
    
    if not ctx.triggered:
        raise PreventUpdate

    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    
    if "training-group-header" in triggered_id and triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
        try:
            import json
            parsed_id = json.loads(triggered_id.split('.')[0])
            selected_group = parsed_id["index"]
            return selected_group, None  
                
        except Exception as e:
            raise PreventUpdate
    else:
        pass

    raise PreventUpdate

@app.callback(
    [Output("training-selected-keyword", "data", allow_duplicate=True),
     Output("training-selected-group", "data", allow_duplicate=True)],
    [Input({"type": "training-select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def select_training_keyword_from_group(n_clicks, display_mode, group_order):

    ctx = dash.callback_context
    
    
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    
    if "training-select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                
                
                keyword_docs = []
                
                if 'df' in globals():
                    for i, text in enumerate(df.iloc[:, 1]):
                        if contains_keyword_word_boundary(str(text), keyword):
                            keyword_docs.append(i)
                    
                    
                    return keyword, None
                else:
                    return keyword, None
            else:
                raise PreventUpdate
            
        except Exception as e:
            raise PreventUpdate
    else:
        pass

    raise PreventUpdate


@app.callback(
    Output("training-articles-container", "children"),
    [Input("training-selected-keyword", "data"),
     Input("training-selected-group", "data"),
     Input("display-mode", "data")],  
    [State("group-order", "data")],
    prevent_initial_call=False  
)
def display_training_recommended_articles(selected_keyword, selected_group, display_mode, group_order):

    
    if display_mode != "training":
        raise PreventUpdate
    
    
    try:
        global df, _ARTICLES_CACHE
        if 'df' not in globals():
            return html.P("Data not loaded")
        
        cache_key = None
        if selected_keyword:
            cache_key = f"training_keyword:{selected_keyword}"
        elif selected_group and group_order:
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    cache_key = f"training_group:{group_name}:{':'.join(sorted(keywords))}"
                    break
        user_data_mtime = get_keysi_user_data_mtime()
        if cache_key and user_data_mtime:
            cache_key = f"{cache_key}:ud:{user_data_mtime}"
        
        if cache_key and cache_key in _ARTICLES_CACHE:
            return _ARTICLES_CACHE[cache_key]
        
        search_keywords = []
        search_title = ""
        skip_group_keyword_resolution = False
        use_snapshot = False
        snapshot_after = get_latest_training_snapshot("after")
        deduped_group_docs = None
        deduped_keyword_matches_in_group = None
        if snapshot_after and snapshot_after.get("group_docs"):
            deduped_group_docs = dedupe_group_docs_by_priority(snapshot_after.get("group_docs", {}), group_order)
            if snapshot_after.get("keyword_matches_in_group"):
                deduped_keyword_matches_in_group = filter_keyword_matches_in_group(
                    snapshot_after.get("keyword_matches_in_group", {}), deduped_group_docs
                )
        
        if selected_keyword:
            search_keywords = [selected_keyword]
            search_title = f"Training Articles containing '{selected_keyword}'"
            if deduped_keyword_matches_in_group is not None:
                keyword_group = resolve_group_for_keyword(group_order, selected_keyword)
                if keyword_group:
                    filtered_indices = deduped_keyword_matches_in_group.get(keyword_group, {}).get(selected_keyword, [])
                    use_filtered_mode = True
                    use_snapshot = True
                    search_title = f"Training Articles containing '{selected_keyword}' (training after snapshot)"
        elif selected_group:

            if selected_group == "Exclude":
                search_title = "Training Articles in Exclude group"
                skip_group_keyword_resolution = True
            if deduped_group_docs is not None:
                filtered_indices = deduped_group_docs.get(selected_group, [])
                use_filtered_mode = True
                use_snapshot = True
                search_title = f"Training Articles in {selected_group} (training after snapshot)"

            if group_order and not skip_group_keyword_resolution:
                search_keywords = []
                for group_name, keywords in group_order.items():
                    if group_name == selected_group:
                        search_keywords = keywords
                        break
                
                if search_keywords:
                    search_title = f"Training Articles containing keywords from group '{selected_group}'"
                else:
                    return html.Div([
                        html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.P(f"Training Group '{selected_group}' has no keywords assigned", 
                               style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                    ])
            elif not skip_group_keyword_resolution:
                return html.Div([
                    html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                    html.P("No training groups have been created yet", 
                           style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                ])
        else:
            return html.Div([
                html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("Please select a training keyword or group to view recommended articles", 
                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
            ])
        
        matching_articles = []
        
        if not use_snapshot:
            filtered_indices = []
            use_filtered_mode = False
        
        if not use_snapshot:
            try:
                filtered_path = FILE_PATHS["filtered_group_assignment"]
                if os.path.exists(filtered_path):
                    with open(filtered_path, "r", encoding="utf-8") as f:
                        filtered_dict = json.load(f)

                    if selected_group in filtered_dict:
                        filtered_indices = filtered_dict[selected_group]
                        use_filtered_mode = True
   
                    elif selected_keyword and group_order:
         
                        keyword_group = None
                        for grp_name, keywords in group_order.items():
                            if selected_keyword in keywords:
                                keyword_group = grp_name
                                break
                    
                        if keyword_group and keyword_group in filtered_dict:
             
                            group_filtered_docs = filtered_dict[keyword_group]
                        
           
                            for idx in group_filtered_docs:
                                if idx < len(df):
                                    text = str(df.iloc[idx, 1])
                                    if contains_keyword_word_boundary(text, selected_keyword):
                                        filtered_indices.append(idx)
                        
                            use_filtered_mode = True
                        else:
                            return html.Div([
                                html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                                html.P(f"Keyword '{selected_keyword}' group not found in training results", 
                                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                            ])
                    else:
                        return html.Div([
                            html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                            html.P(f"Keyword '{selected_keyword}' group information not available", 
                                   style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                        ])
                else:
                    return html.Div([
                        html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.P("No training results available. Please run training first.", 
                               style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                    ])
            except Exception as e:
                return html.Div([
                    html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                    html.P(f"Error loading training results: {str(e)}", 
                           style={"color": "#e74c3c", "textAlign": "center", "padding": "20px"})
                ])
        
        if use_filtered_mode:
            for idx in filtered_indices:
                if idx < len(df):
                    row = df.iloc[idx]
                    text = str(row.iloc[1]) if len(row) > 1 else ""
                    file_keywords = extract_top_keywords(text, 5)
                    matching_articles.append({
                        'file_number': idx + 1,
                        'file_index': idx,
                        'text': text,
                        'keywords': file_keywords
                    })
        else:
        
            has_short_keyword = any(len(kw) <= 2 for kw in search_keywords)
            
            if has_short_keyword:
                for kw in search_keywords:
                    for i in range(len(df)):
                        text = str(df.iloc[i, 1])
                        if contains_keyword_word_boundary(text, kw):
                            if not any(article['file_index'] == i for article in matching_articles):
                                file_keywords = extract_top_keywords(str(df.iloc[i, 1]), 5)
                                matching_articles.append({
                                    'file_number': i + 1,
                                    'file_index': i,
                                    'text': str(df.iloc[i, 1]),
                                    'keywords': file_keywords,
                                    'bm25_score': 1.0
                                })
            else:
                try:
                    from rank_bm25 import BM25Okapi
                    from nltk.stem import SnowballStemmer
                    import re
                    
                    stemmer = SnowballStemmer('english')
                    
                    def process_articles_serial(articles):
                        tokenized_corpus, valid_indices = [], []
                        for i, a in enumerate(articles):
                            try:
                                a = re.sub(r'\d+', '', str(a)).strip()
                                if len(a.split()) >= 5:
                                    words = word_tokenize(a)
                                    stemmed = [stemmer.stem(w.lower()) for w in words]
                                    if stemmed:
                                        tokenized_corpus.append(" ".join(stemmed))
                                        valid_indices.append(i)
                            except Exception:
                                pass
                        return tokenized_corpus, valid_indices
                    
                    all_texts = [str(df.iloc[i, 1]) for i in range(len(df))]
                    tokenized_corpus, valid_indices = process_articles_serial(all_texts)
                    bm25 = BM25Okapi([s.split() for s in tokenized_corpus])
                    
                    query_tokens = []
                    for kw in search_keywords:
                        q = [stemmer.stem(w.lower()) for w in word_tokenize(kw)]
                        query_tokens.extend(q)
                    
                    if not query_tokens:
                        for kw in search_keywords:
                            for i in range(len(df)):
                                text = str(df.iloc[i, 1])
                                if contains_keyword_word_boundary(text, kw):
                                    if not any(article['file_index'] == i for article in matching_articles):
                                        file_keywords = extract_top_keywords(str(df.iloc[i, 1]), 5)
                                        matching_articles.append({
                                            'file_number': i + 1,
                                            'file_index': i,
                                            'text': str(df.iloc[i, 1]),
                                            'keywords': file_keywords,
                                            'bm25_score': 1.0
                                        })
                    else:
                        scores = bm25.get_scores(query_tokens)
                        
                        idx_corpus = [i for i, s in enumerate(scores) if s > 0.1]
                        if len(idx_corpus) == 0:
                            idx_corpus = [i for i, s in enumerate(scores) if s > 0.01]
                        
                        score_idx_pairs = [(scores[i], valid_indices[i]) for i in idx_corpus]
                        score_idx_pairs.sort(reverse=True)
                        
                        for score, idx in score_idx_pairs:
                            if idx < len(df):
                                text = str(df.iloc[idx, 1])
                                file_keywords = extract_top_keywords(text, 5)
                                matching_articles.append({
                                    'file_number': idx + 1,
                                    'file_index': idx,
                                    'text': text,
                                    'keywords': file_keywords,
                                    'bm25_score': float(score)
                                })
                                
                                if len(matching_articles) >= 100:
                                    break
                        
                        matching_articles = sorted(matching_articles, key=lambda x: x.get('bm25_score', 0), reverse=True)[:100]
                    
                except Exception as e:
                    traceback.print_exc()
        
        if not matching_articles:
            result = html.P(f"No training articles found for the selected search criteria")
            if cache_key:
                _ARTICLES_CACHE[cache_key] = result
            return result
        
        article_items = [
            html.H6(f"{search_title} (Found {len(matching_articles)} articles)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        
        for article_info in matching_articles:
            keyword_tags = []
            for keyword in article_info['keywords']:
                keyword_tag = html.Span(
                    keyword,
                    style={
                        "backgroundColor": "#e3f2fd",
                        "color": "#1976d2",
                        "padding": "2px 6px",
                        "margin": "2px",
                        "borderRadius": "12px",
                        "fontSize": "11px",
                        "display": "inline-block"
                    }
                )
                keyword_tags.append(keyword_tag)
            
            article_item = html.Div([
                html.Button(
                    html.Div([
                        html.H6(f"Training Article {article_info['file_number']}", 
                               style={"color": "#333", "marginBottom": "8px", "fontSize": "14px", "margin": "0"}),
                        html.Div([
                            html.Span("Top 5 Keywords: ", style={"fontWeight": "bold", "color": "#666"}),
                            html.Div(keyword_tags, style={"display": "inline-block", "marginLeft": "5px"})
                        ], style={"marginBottom": "8px"}),
                    ]),
                    id={"type": "training-article-item", "index": article_info['file_index']},
                    className="article-item-button",
                    style={
                        "width": "100%",
                        "padding": "12px", 
                        "border": "1px solid #eee",
                        "backgroundColor": "white",
                        "borderRadius": "5px",
                        "marginBottom": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                        "cursor": "pointer",
                        "textAlign": "left",
                        "outline": "none"
                    },
                    n_clicks=0
                ),
                html.Hr(style={"margin": "4px 0", "borderColor": "#ddd"})
            ])
            article_items.append(article_item)
        
        result = html.Div(article_items)
        if cache_key:
            _ARTICLES_CACHE[cache_key] = result
        
        return result
        
    except Exception as e:
        return html.P(f"Error displaying training recommended articles: {str(e)}")

@app.callback(
    [Output("training-article-fulltext-container", "children"),
     Output("training-selected-article", "data")],
    [Input({"type": "training-article-item", "index": ALL}, "n_clicks")],
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def display_training_article_content(article_clicks, display_mode):

    ctx = dash.callback_context
    
    
    if not ctx.triggered:
        raise PreventUpdate
    
    article_index = None
    if 'training-article-item' in ctx.triggered[0]['prop_id']:
        try:
            triggered_id = ctx.triggered[0]['prop_id']
            btn_info = json.loads(triggered_id.split('.')[0])
            article_index = btn_info.get("index")
        except Exception as e:
            raise PreventUpdate
    
    if article_index is None:
        raise PreventUpdate
    
    try:
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        if article_index is not None and article_index < len(df):
            article_text = str(df.iloc[article_index, 1])
            
            content = html.Div([
                html.H5(f"Training Article {article_index + 1}", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
            
            
            
            return content, article_index
            
        else:
            return html.P("Training article not found", style={"color": "red"}), None
    
    except Exception as e:
        return html.P(f"Error loading training article: {str(e)}", style={"color": "red"}), None





@app.callback(
    [Output("article-fulltext-container", "children"),
     Output("selected-article", "data", allow_duplicate=True)],
    [Input({"type": "article-item", "index": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("selected-keyword", "data"),
     State("selected-group", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def display_article_content_smart(article_clicks, display_mode, current_keyword, current_group, group_order):

    ctx = dash.callback_context
    
    
    if not ctx.triggered:
        raise PreventUpdate
    
    article_index = None
    if 'article-item' in ctx.triggered[0]['prop_id']:
        try:
            import json
            triggered_id = ctx.triggered[0]['prop_id']
            btn_info = json.loads(triggered_id.split('.')[0])
            article_index = btn_info.get("index")
        except Exception as e:
            raise PreventUpdate
    
    if article_index is None:
        raise PreventUpdate
    
    try:
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        if article_index is not None and article_index < len(df):
            article_text = str(df.iloc[article_index, 1])
            
            content = html.Div([
                html.H5(f"Article {article_index + 1}", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
            
            
            
            return content, article_index
            
        else:
            return html.P("Article not found", style={"color": "red"}), None
    
    except Exception as e:
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), None


@app.callback(
    [Output("display-mode", "data", allow_duplicate=True),
     Output("switch-finetune-btn", "children")],
    Input("switch-finetune-btn", "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def switch_to_finetune_mode(n_clicks, current_mode):

    if not n_clicks:
        raise PreventUpdate
    if current_mode == "training":
        return "finetune", "Switch to Training View"
    if current_mode == "finetune":
        return "training", "Switch to Finetune Mode"
    raise PreventUpdate

@app.callback(
    Output("finetune-group-containers", "children"),
    [Input("group-order", "data"),
     Input("finetune-selected-group", "data"),  
     Input("finetune-selected-keyword", "data")]  
)
def render_finetune_groups(group_order, selected_group, selected_keyword):

    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
        if grp_name == "Exclude":

            if kw_list:  
                group_display_name = "Exclude"
                group_color = get_group_color(grp_name)
            else:  
                group_display_name = "Other (Exclude)"
                group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Group {group_number}"
            group_color = get_group_color(grp_name)
        
        if grp_name == "Exclude":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  
            }
        else:
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px solid {group_color}",
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px"
            }
        
        group_header = html.Button(
            group_display_name,
            id={"type": "finetune-group-header", "index": grp_name},
            style=header_style
        )

        group_keywords = []
        for kw in kw_list:
            is_selected = selected_keyword and kw == selected_keyword
            
            keyword_button = html.Button(
                kw,
                id={"type": "finetune-select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  
                    "color": "white" if is_selected else group_color,  
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  
                }
            )
            group_keywords.append(keyword_button)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "8px",
                "minHeight": "50px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "10px"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"marginBottom": "15px"}
        )
        children.append(group_container)

    return children

@app.callback(
    [Output("finetune-selected-group", "data"),
     Output("finetune-selected-keyword", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    Input({"type": "finetune-group-header", "index": ALL}, "n_clicks"),
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def select_finetune_group(n_clicks, display_mode):
    ctx = dash.callback_context
    
    
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        triggered_value = ctx.triggered[0]['value']
    
    if display_mode != "finetune":
        raise PreventUpdate
    
    if not ctx.triggered:
        raise PreventUpdate
    
    trig = ctx.triggered[0]['prop_id']
    trig_value = ctx.triggered[0]['value']
    
    
    if "finetune-group-header" in trig:
        if trig_value and (isinstance(trig_value, (int, float)) and trig_value > 0):
            try:
                info = json.loads(trig.split('.')[0])
                group_name = info.get("index")
                
                try:
                    user_finetuned_path = FILE_PATHS["user_finetuned_list"]
                    filtered_path = FILE_PATHS["filtered_group_assignment"]
                    bm25_path = FILE_PATHS["bm25_search_results"]
                    
                    loaded_from = None
                    matched_dict = None
                    
                    if os.path.exists(user_finetuned_path):
                        with open(user_finetuned_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "user_finetuned_list.json"
                    elif os.path.exists(filtered_path):
                        with open(filtered_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "filtered_group_assignment.json"
                    elif os.path.exists(bm25_path):
                        with open(bm25_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "bm25_search_results.json"
                    
                    if matched_dict:
                        if group_name in matched_dict:
                            doc_count = len(matched_dict[group_name])
                            if doc_count <= 20:
                                pass
                            else:
                                pass

                        total = 0
                        for grp, indices in matched_dict.items():
                            count = len(indices) if isinstance(indices, list) else 0
                            total += count
                except Exception as e:
                    pass

                return group_name, None, None  
            except Exception as e:
                raise PreventUpdate
        else:
            pass

    raise PreventUpdate

@app.callback(
    [Output("finetune-selected-keyword", "data", allow_duplicate=True),
     Output("finetune-selected-group", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    [Input({"type": "finetune-select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def select_finetune_keyword_from_group(n_clicks, display_mode):

    ctx = dash.callback_context
    
    
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        triggered_value = ctx.triggered[0]['value']
    
    if display_mode != "finetune":
        raise PreventUpdate
    
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    if "finetune-select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            group = btn_info.get("group")  
            
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                
                try:
                    user_finetuned_path = FILE_PATHS["user_finetuned_list"]
                    filtered_path = FILE_PATHS["filtered_group_assignment"]
                    bm25_path = FILE_PATHS["bm25_search_results"]
                    
                    loaded_from = None
                    matched_dict = None
                    
                    if os.path.exists(user_finetuned_path):
                        with open(user_finetuned_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "user_finetuned_list.json"
                    elif os.path.exists(filtered_path):
                        with open(filtered_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "filtered_group_assignment.json"
                    elif os.path.exists(bm25_path):
                        with open(bm25_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "bm25_search_results.json"
                    
                    if matched_dict:
                        if group in matched_dict:
                            group_docs = matched_dict[group]
                            
                            try:
                                df_local = pd.read_csv(FILE_PATHS["csv_path"])
                                keyword_doc_count = 0
                                keyword_doc_indices = []
                                for idx in group_docs:
                                    if idx < len(df_local):
                                        text = str(df_local.iloc[idx, 1])
                                        if contains_keyword_word_boundary(text, keyword):
                                            keyword_doc_count += 1
                                            keyword_doc_indices.append(idx)
                                
                                if keyword_doc_count <= 20:
                                    pass
                            except Exception as e:
                                pass
                        else:
                            pass
                except Exception as e:
                    pass

                return keyword, group, None  
        except Exception as e:
            raise PreventUpdate
    
    raise PreventUpdate

@app.callback(
    Output("finetune-articles-container", "children"),
    [Input("finetune-selected-group", "data"),
     Input("finetune-selected-keyword", "data"),
     Input("finetune-highlight-core", "data"),
     Input("finetune-selected-article-index", "data")],  
    [State("group-order", "data"),
     State("display-mode", "data")]
)
def display_finetune_articles(selected_group, selected_keyword, core_indices, selected_article_idx, group_order, display_mode):

    if display_mode != "finetune":
        raise PreventUpdate
    
    if not selected_group and not selected_keyword:
        return html.P("Select a group or keyword to view documents", 
                     style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "40px 20px"})
    
    
    try:
        global df
        if 'df' not in globals():
            return html.P("Data not loaded", style={"color": "#e74c3c", "textAlign": "center"})
        
        doc_indices = []
        doc_types = {}
        
        if core_indices:
            for idx in core_indices:
                doc_indices.append(idx)
                doc_types[idx] = "core"
        
        if not doc_indices:
            return html.P("No documents to display", 
                         style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
        
        articles = []
        for i, idx in enumerate(sorted(doc_indices)):
            if idx >= len(df):
                continue
            
            text = str(df.iloc[idx, 1])[:200] + "..." if len(str(df.iloc[idx, 1])) > 200 else str(df.iloc[idx, 1])
            
            doc_type = doc_types.get(idx, "background")
            
            
            if doc_type == "core":
                type_label = ""
                type_color = "#FFD700"
                border_color = "#FFD700"
            else:
                type_label = "        Background"
                type_color = "#1f77b4"
                border_color = "#1f77b4"
            
            is_selected = (selected_article_idx is not None and selected_article_idx == idx)
            
            if is_selected:
                card_style = {
                    "padding": "12px",
                    "marginBottom": "10px",
                    "borderRadius": "6px",
                    "border": "2px solid #FF4444",  
                    "backgroundColor": "white",
                    "cursor": "pointer",
                    "transition": "all 0.2s ease",
                    "boxShadow": "0 2px 6px rgba(255, 68, 68, 0.2)",  
                    "scrollMarginTop": "100px"
                }
                doc_label_style = {"fontWeight": "bold", "marginRight": "10px", "color": "#FF4444"}  
            else:
                card_style = {
                    "padding": "12px",
                    "marginBottom": "10px",
                    "borderRadius": "6px",
                    "border": f"1px solid {border_color}",
                    "backgroundColor": "white",
                    "cursor": "pointer",
                    "transition": "all 0.2s ease",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"
                }
                doc_label_style = {"fontWeight": "bold", "marginRight": "10px", "color": "#2c3e50"}
            
            article_card = html.Div([
                html.Div([
                    html.Span(f"Doc {idx+1}", style=doc_label_style),  
                    html.Span(type_label, style={"fontSize": "0.85rem", "color": type_color, "fontWeight": "bold"})
                ], style={"marginBottom": "8px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                html.P(text, style={"color": "#34495e", "fontSize": "0.9rem", "margin": "0", "lineHeight": "1.4"})
            ], id={"type": "finetune-article-card", "index": idx}, className="finetune-doc-card", style=card_style)
            articles.append(article_card)
        
        title_text = ""
        if selected_keyword and selected_group:
            title_text = f"Showing {len(doc_indices)} documents for '{selected_keyword}' in {selected_group}"
        elif selected_keyword:
            title_text = f"Showing {len(doc_indices)} documents for '{selected_keyword}'"
        elif selected_group:
            title_text = f"Showing {len(doc_indices)} documents in {selected_group}"
        
        return [
            html.P(title_text, style={"color": "#2c3e50", "fontWeight": "bold", "marginBottom": "15px", "textAlign": "center", "fontSize": "0.95rem"}),
            html.Div(articles)
        ]
        
    except Exception as e:
           
        traceback.print_exc()
        return html.P(f"Error: {str(e)}", style={"color": "#e74c3c", "textAlign": "center"})

@app.callback(
    Output("finetune-selected-article-index", "data", allow_duplicate=True),
    Input({"type": "finetune-article-card", "index": ALL}, "n_clicks"),
    [State("display-mode", "data"),
     State("finetune-selected-article-index", "data")],
    prevent_initial_call=True
)
def handle_finetune_article_click(n_clicks, display_mode, current_selected):

    if display_mode != "finetune":
        raise PreventUpdate
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_value = ctx.triggered[0]['value']
    
    if not triggered_value or triggered_value == 0:
        raise PreventUpdate
    
    try:
      
        card_info = json.loads(triggered_id.split('.')[0])
        article_idx = card_info.get("index")
        
        
        return article_idx
        
    except Exception as e:
           
        traceback.print_exc()
        return current_selected

@app.callback(
    Output("finetune-text-container", "children"),
    Input("finetune-selected-article-index", "data"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def update_finetune_text_preview(selected_idx, display_mode):

    if display_mode != "finetune":
        raise PreventUpdate
    
    if selected_idx is None:
        return html.P("Click a document to preview", 
                     style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "20px", "fontSize": "0.9rem"})
    
    try:
        global df
        if 'df' not in globals() or selected_idx >= len(df):
            return html.P("Document not found", style={"color": "#e74c3c"})
        
        full_text = str(df.iloc[selected_idx, 1])
        
        preview = html.Div([
            html.H5(f"Document {selected_idx+1}", style={"color": "#2c3e50", "marginBottom": "10px", "fontSize": "1rem"}),
            html.P(full_text, style={
                "color": "#34495e", 
                "fontSize": "0.85rem", 
                "lineHeight": "1.5",
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word"
            })
        ])
        
        return preview
        
    except Exception as e:
           
        traceback.print_exc()
        return html.P(f"Error: {str(e)}", style={"color": "#e74c3c"})

@app.callback(
    [Output("finetune-highlight-core", "data"),
     Output("finetune-operation-buttons", "children")],  
    [Input("finetune-selected-group", "data"),
     Input("finetune-selected-keyword", "data"),
     Input("finetune-selected-article-index", "data")],  
    [State("group-order", "data"),
     State("finetune-temp-assignments", "data")]
)
def compute_finetune_highlights(selected_group, selected_keyword, selected_article_idx, group_order, temp_assignments):
    global df, current_group_order
    core = []
    operation_buttons = []
    
    current_group_order = group_order
    
    for grp_name, kw_list in group_order.items():
        pass

    snapshot_refine = get_latest_refinement_snapshot()
    snapshot_after = get_latest_training_snapshot("after")
    snapshot_dict = None
    snapshot_groups = None
    snapshot_keyword_matches = None
    if snapshot_refine and isinstance(snapshot_refine, dict):
        refine_groups = snapshot_refine.get("group_docs") or {}
        if (selected_group and selected_group in refine_groups) or refine_groups:
            snapshot_dict = snapshot_refine
            snapshot_groups = refine_groups
    if snapshot_dict is None and snapshot_after and isinstance(snapshot_after, dict):
        snapshot_dict = snapshot_after
        snapshot_groups = snapshot_after.get("group_docs")
    if snapshot_groups:
        snapshot_groups = dedupe_group_docs_by_priority(snapshot_groups, group_order)
        if snapshot_dict and snapshot_dict.get("keyword_matches_in_group"):
            snapshot_keyword_matches = filter_keyword_matches_in_group(
                snapshot_dict.get("keyword_matches_in_group", {}), snapshot_groups
            )
    
    excluded_group = None
    if selected_article_idx is not None and group_order:
        try:
            matched_dict = None
            if snapshot_groups:
                matched_dict = snapshot_groups
            else:
                matched_dict_path = FILE_PATHS["filtered_group_assignment"]
                if os.path.exists(matched_dict_path):
                    with open(matched_dict_path, "r", encoding="utf-8") as f:
                        matched_dict = json.load(f)
                
            if matched_dict:
                for grp_name in matched_dict.keys():
                    if isinstance(matched_dict[grp_name], list) and len(matched_dict[grp_name]) > 0:
                        if isinstance(matched_dict[grp_name][0], str):
                            matched_dict[grp_name] = [int(x) for x in matched_dict[grp_name]]
                
                for grp_name, indices in matched_dict.items():
                    try:
                        if isinstance(indices, list):
                            int_indices = []
                            for idx in indices:
                                try:
                                    int_indices.append(int(idx))
                                except (ValueError, TypeError):
                                    continue
                            
                            try:
                                selected_idx = int(selected_article_idx)
                                if selected_idx in int_indices:
                                    excluded_group = grp_name
                                    break
                            except (ValueError, TypeError):
                                continue
                    except Exception as e:
                        continue
        except Exception as e:
            pass

    if temp_assignments and selected_article_idx is not None:
        for idx_str, target_group in temp_assignments.items():
            if idx_str.endswith("_original"):
                continue
            try:
                if int(idx_str) == int(selected_article_idx):
                    excluded_group = target_group
                    break
            except (ValueError, TypeError):
                continue
    
    
    if group_order and selected_article_idx is not None:
        button_style_base = {
            "color": "white",
            "border": "none",
            "padding": "10px 16px",
            "borderRadius": "6px",
            "fontSize": "0.95rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "width": "100%",
            "marginBottom": "8px"
        }
        
        for group_name in group_order.keys():

            should_exclude = False
            if excluded_group is not None:
                if group_name == excluded_group:
                    should_exclude = True
               
            
            if should_exclude:
                continue
                

            if group_name == "Group 1":
                bg_color = "#FF6B6B"  
            elif group_name == "Group 2":
                bg_color = "#32CD32"  
            elif group_name == "Exclude":
                bg_color = "#e74c3c"  
            else:
                bg_color = "#3498db"  
            
            button_style = {**button_style_base, "backgroundColor": bg_color}
            

            if group_name == "Exclude" and group_order.get("Exclude"):
                button_text = "Move to Exclude"
            else:
                button_text = f"Move to {group_name}"
            
            operation_buttons.append(
                html.Button(
                    button_text,
                    id={"type": "finetune-move-to", "target": group_name},
                    n_clicks=0,
                    style=button_style
                )
            )
        
    else:
        operation_buttons = [
            html.P("Select a document to perform operations", 
                   style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "10px"})
        ]
    
    
    if not selected_group or not group_order or selected_group not in group_order:
        if selected_keyword and selected_keyword.strip():
            pass
        return core, operation_buttons
    
    try:
        matched_dict_path = None
        is_already_filtered = False
        
        if snapshot_groups:
            matched_dict = snapshot_groups
        else:
            if os.path.exists(FILE_PATHS["filtered_group_assignment"]):
                matched_dict_path = FILE_PATHS["filtered_group_assignment"]
                is_already_filtered = True
            elif os.path.exists(FILE_PATHS["bm25_search_results"]):
                matched_dict_path = FILE_PATHS["bm25_search_results"]
                is_already_filtered = False
            else:
                return core, operation_buttons
            
            
            with open(matched_dict_path, "r", encoding="utf-8") as f:
                matched_dict = json.load(f)
        
        for grp_name in matched_dict.keys():
            if isinstance(matched_dict[grp_name], list) and len(matched_dict[grp_name]) > 0:
                if isinstance(matched_dict[grp_name][0], str):
                    matched_dict[grp_name] = [int(x) for x in matched_dict[grp_name]]
        
        for grp_name, indices in matched_dict.items():
            if len(indices) <= 15:
                pass

        if temp_assignments:
            for idx_str, target_group in temp_assignments.items():
                if not idx_str.endswith("_original"):
                    pass

            for idx_str, target_group in temp_assignments.items():
                if idx_str.endswith("_original"):
                    continue
                
                try:
                    idx = int(idx_str)
                    
                    removed_from = None
                    for grp_name in matched_dict.keys():
                        if idx in matched_dict[grp_name]:
                            matched_dict[grp_name].remove(idx)
                            removed_from = grp_name
                            break
                    
                    if target_group in matched_dict:
                        matched_dict[target_group].append(idx)
                    elif target_group not in matched_dict:
                        matched_dict[target_group] = [idx]
                        
                except Exception as e:
                    pass

            for grp_name, indices in matched_dict.items():
                if len(indices) <= 15:
                    pass
        else:
            pass

        if selected_group == "Exclude":
            exclude_has_keywords = False
            if group_order and "Exclude" in group_order and group_order["Exclude"]:
                exclude_has_keywords = True
            else:
                pass

            try:
                if exclude_has_keywords:
                   
                    exclude_keywords = group_order.get("Exclude", [])
                    
                    
                    filtered_exclude_indices = []
                    if "Exclude" in matched_dict:
                        filtered_exclude_indices = matched_dict["Exclude"]
                    
                   
                    manually_moved_indices = []
                    if temp_assignments:
                        for idx_str, target_group in temp_assignments.items():
                            if idx_str.endswith("_original"):
                                continue
                            if target_group == "Exclude":  
                                manually_moved_indices.append(int(idx_str))
                    
                    all_exclude_indices = list(set(filtered_exclude_indices + manually_moved_indices))
                    
                    return all_exclude_indices, operation_buttons
                else:
                    filtered_exclude_indices = []
                    if "Exclude" in matched_dict:
                        filtered_exclude_indices = matched_dict["Exclude"]
                    
                    manually_moved_indices = []
                    if temp_assignments:
                        for idx_str, target_group in temp_assignments.items():
                            if idx_str.endswith("_original"):
                                continue
                            if target_group == "Exclude":  
                                manually_moved_indices.append(int(idx_str))
                    
                    all_exclude_indices = list(set(filtered_exclude_indices + manually_moved_indices))
                    return all_exclude_indices, operation_buttons
            except Exception as e:
                exclude_indices = matched_dict.get("Exclude", [])
                return exclude_indices, operation_buttons

        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        
        if selected_group in matched_dict:
            selected_group_indices = matched_dict[selected_group].copy()
            if len(selected_group_indices) <= 20:
                pass
        else:
            selected_group_indices = []
        
        if temp_assignments:
            for idx_str, target_group in temp_assignments.items():
                if idx_str.endswith("_original"):
                    continue
                try:
                    idx = int(idx_str)
                    if target_group == selected_group:
                        if idx not in selected_group_indices:
                            selected_group_indices.append(idx)
                    else:
                        if idx in selected_group_indices:
                            selected_group_indices.remove(idx)
                except Exception as e:
                    pass

        if selected_keyword and selected_keyword.strip():
            keyword_matched_indices = None
            if snapshot_keyword_matches is not None:
                keyword_matched_indices = snapshot_keyword_matches.get(selected_group, {}).get(selected_keyword, None)
            if keyword_matched_indices is None:
                keyword_matched_indices = []
                for idx in selected_group_indices:
                    if idx >= len(df):
                        continue
                    text = str(df.iloc[idx, 1])
                    if contains_keyword_word_boundary(text, selected_keyword):
                        keyword_matched_indices.append(idx)
            selected_group_indices = keyword_matched_indices
        
        
        return selected_group_indices, operation_buttons
        
    except Exception as e:
        traceback.print_exc()
        return core, operation_buttons


@app.callback(
    Output('finetune-2d-plot', 'figure'),
    [Input('display-mode', 'data'),
     Input('finetune-highlight-core', 'data'),
     Input('finetune-selected-article-index', 'data')],  
    [State('training-figures', 'data'),
     State('finetune-figures', 'data'), 
     State('finetune-selected-group', 'data')]
)
def render_finetune_plot(display_mode, core_indices, selected_article_idx, training_figures, finetune_figures, selected_group):
    if display_mode != "finetune":
        raise PreventUpdate
    

    active_figure = training_figures.get('after') if training_figures else None
    
   
    idx_to_coord = {}
    
    try:
        if isinstance(active_figure, dict):
            after = active_figure
            if after.get('data') and len(after['data']) > 0:
                
               
                for trace in after['data']:
                    trace_x = trace.get('x', [])
                    trace_y = trace.get('y', [])
                    trace_customdata = trace.get('customdata', [])
                    
                    if trace_x and trace_y:
                        for i, (x, y) in enumerate(zip(trace_x, trace_y)):
                           
                            if i < len(trace_customdata) and trace_customdata[i]:
                                doc_idx = trace_customdata[i][0] if isinstance(trace_customdata[i], list) else trace_customdata[i]
                                idx_to_coord[doc_idx] = (x, y)
                
    except Exception as e:
           
        traceback.print_exc()


    if not idx_to_coord:
        tsne_result = get_document_tsne()
        if tsne_result is not None:
            coords = tsne_result
            idx_to_coord = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}
        else:

            return {
                'data': [],
                'layout': {'title': 'Finetune - No coordinates available'}
            }


    valid_indices = list(idx_to_coord.keys())
    if valid_indices:
        pass

    all_idx = set(valid_indices)
    core_set = set(core_indices or []) & all_idx
    
    
    all_other_idx = list(all_idx - core_set)
    

    traces = []

    if isinstance(training_figures, dict) and isinstance(training_figures.get('after'), dict):
        after_data = training_figures['after'].get('data', [])
        for trace in after_data:
           
            traces.append(trace.copy())
    

    if core_set:
        cidx = list(core_set)
        core_style = PLOT_STYLES["core"]
        traces.append({
            'x': [idx_to_coord[i][0] for i in cidx],
            'y': [idx_to_coord[i][1] for i in cidx],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Selected Group',
            'marker': {
                'size': core_style["size"],
                'color': core_style["color"],
                'opacity': core_style["opacity"],
                'symbol': core_style["symbol"],
                'line': {'width': core_style["line_width"], 'color': core_style["line_color"]}
            },
            'text': [f'Doc {i+1}' for i in cidx],
            'customdata': [[i] for i in cidx],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        })
    

    if selected_article_idx is not None and selected_article_idx in idx_to_coord:
        traces.append({
            'x': [idx_to_coord[selected_article_idx][0]],
            'y': [idx_to_coord[selected_article_idx][1]],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Selected Document',
            'marker': {
                'size': 16,  
                'color': '#FF4444',  
                'opacity': 1.0,
                'symbol': 'star',
                'line': {'width': 2, 'color': '#FF0000'} 
            },
            'text': [f'Doc {selected_article_idx+1} (Selected)'],
            'customdata': [[selected_article_idx]],
            'hovertemplate': '<b>%{text}</b><extra></extra>',
            'showlegend': True
        })

    
   
    layout_style = PLOT_STYLES["layout"]
    
    if isinstance(training_figures, dict) and isinstance(training_figures.get('after'), dict):
        
        import copy
        base_layout = copy.deepcopy(training_figures['after'].get('layout', {}))
        
        
        if 'title' in base_layout:
            if isinstance(base_layout['title'], dict):
                base_layout['title']['text'] = 'Finetune Mode - Interactive 2D Visualization'
            else:
                base_layout['title'] = {
                    'text': 'Finetune Mode - Interactive 2D Visualization',
                    'font': {'size': 18, 'color': '#2c3e50'},
                    'x': 0.5
                }
        else:
            base_layout['title'] = {
                'text': 'Finetune Mode - Interactive 2D Visualization',
                'font': {'size': 18, 'color': '#2c3e50'},
                'x': 0.5
            }
        
       
        base_layout['xaxis'] = {
            **base_layout.get('xaxis', {}),
            **layout_style["xaxis"],
            'title': base_layout.get('xaxis', {}).get('title', 'X')  
        }
        base_layout['yaxis'] = {
            **base_layout.get('yaxis', {}),
            **layout_style["yaxis"],
            'title': base_layout.get('yaxis', {}).get('title', 'Y') 
        }
        base_layout['plot_bgcolor'] = layout_style["plot_bgcolor"]
        base_layout['paper_bgcolor'] = layout_style["paper_bgcolor"]
    else:
       
        base_layout = {
            'title': {
                'text': 'Finetune Mode - Interactive 2D Visualization',
                'font': {'size': 18, 'color': '#2c3e50'},
                'x': 0.5
            },
            'xaxis': {**layout_style["xaxis"], 'title': 'X'},
            'yaxis': {**layout_style["yaxis"], 'title': 'Y'},
            'hovermode': 'closest',
            'showlegend': True,
            'plot_bgcolor': layout_style["plot_bgcolor"],
            'paper_bgcolor': layout_style["paper_bgcolor"],
            'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50}
        }
    
    fig = {
        'data': traces,
        'layout': base_layout
    }
    return fig


@app.callback(
    Output("finetune-selected-article-index", "data", allow_duplicate=True),
    Input('finetune-2d-plot', 'clickData'),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def finetune_click_2d_point(click_data, display_mode):

    if display_mode != "finetune":
        raise PreventUpdate
    
    if not click_data:
        raise PreventUpdate
    
    try:
        point = click_data['points'][0]
        customdata = point.get('customdata')
        if isinstance(customdata, list) and len(customdata) > 0:
            idx = customdata[0]
        elif customdata is not None:
            idx = customdata
        else:
            idx = point.get('pointIndex')
        if idx is None:
            raise PreventUpdate
        idx = int(idx)
        return idx
    except Exception as e:
        raise PreventUpdate


@app.callback(
    [Output("finetune-temp-assignments", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    Input({"type": "finetune-move-to", "target": ALL}, "n_clicks"),
    [State("finetune-selected-article-index", "data"),
     State("finetune-temp-assignments", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def finetune_move_document(n_clicks_list, selected_idx, assignments, group_order):

    
    if not dash.callback_context.triggered:
        raise PreventUpdate
    
    triggered = dash.callback_context.triggered[0]
    
    if selected_idx is None:
        raise PreventUpdate
    
  
    triggered_id = triggered['prop_id']
    triggered_value = triggered['value']
    
    
    if triggered_id == '.':
        raise PreventUpdate
    
  
    if triggered_value is None or triggered_value == 0:
        raise PreventUpdate
    
    try:
    
        
        button_id = json_module.loads(triggered_id.split('.')[0])
        target_group = button_id['target']
        
        new_map = dict(assignments or {})

        if str(selected_idx) not in assignments:
            try:
                original_group = "Unknown"
                snapshot_for_training = get_latest_snapshot_for_training()
                if snapshot_for_training and snapshot_for_training.get("group_docs"):
                    current_matched_dict = snapshot_for_training.get("group_docs", {})
                    for grp_name_file, indices in current_matched_dict.items():
                        if selected_idx in indices:
                            original_group = grp_name_file
                            break
                new_map[f"{selected_idx}_original"] = original_group
            except Exception as e:
                new_map[f"{selected_idx}_original"] = "Unknown"
        
        new_map[str(selected_idx)] = target_group
        
        # 使用 keysi_user_data 快照更新并记录变更（不再使用 filtered_group_assignment.json）
        original_group_name = None
        try:
            snapshot_for_training = get_latest_snapshot_for_training()
            current_matched_dict = {}
            if snapshot_for_training and snapshot_for_training.get("group_docs"):
                current_matched_dict = {g: list(idxs) for g, idxs in snapshot_for_training.get("group_docs", {}).items()}
            for g in (group_order or {}).keys():
                if g not in current_matched_dict:
                    current_matched_dict[g] = []
            for grp_name in current_matched_dict.keys():
                if selected_idx in current_matched_dict[grp_name]:
                    original_group_name = grp_name
                    current_matched_dict[grp_name].remove(selected_idx)
                    break
            if original_group_name == target_group:
                return new_map, None
            if target_group not in current_matched_dict:
                current_matched_dict[target_group] = []
            if selected_idx not in current_matched_dict[target_group]:
                current_matched_dict[target_group].append(selected_idx)

            doc_preview = None
            try:
                if 'df' in globals() and df is not None and selected_idx < len(df):
                    text = str(df.iloc[selected_idx, 1])
                    doc_preview = text[:120] + "..." if len(text) > 120 else text
            except Exception:
                doc_preview = None

            refinement_change = {
                "document_index": selected_idx,
                "from_group": original_group_name if original_group_name else "Unknown",
                "to_group": target_group,
                "group_counts_after": {g: len(indices) for g, indices in current_matched_dict.items()}
            }
            if doc_preview:
                refinement_change["doc_preview"] = doc_preview
            record_user_data(
                "refinement_change",
                group_order=group_order,
                matched_dict=current_matched_dict,
                refinement_change=refinement_change
            )
        except Exception as e:
            pass

        return new_map, None
    except Exception as e:
           
        traceback.print_exc()
        raise PreventUpdate


@app.callback(
    Output("finetune-temp-assignments", "data", allow_duplicate=True),
    Input("finetune-clear-history-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_finetune_history(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    try:
        user_data = load_keysi_user_data()
        if not user_data:
            user_data = {"training_sessions": [], "refinement_changes": []}
        user_data["refinement_changes"] = []
        write_json_atomic(get_user_data_path(), user_data)
    except Exception as e:
        pass
    return {}


@app.callback(
    [Output("finetune-adjustment-history", "children"),
     Output("finetune-history-buttons", "children")],
    [Input("finetune-temp-assignments", "data")]
)
def update_adjustment_history(temp_assignments):
    global df, current_group_order
    user_data = load_keysi_user_data()
    changes = user_data.get("refinement_changes", []) if user_data else []
    if not changes:
        return html.P("No adjustments yet. Click a point and use the buttons to reassign.", 
                     style={
                         "color": "#7f8c8d", 
                         "fontStyle": "italic", 
                         "textAlign": "center", 
                         "padding": "20px",
                         "fontSize": "0.95rem"
                     }), []
    
    if 'df' not in globals():
        try:
            df = pd.read_csv(FILE_PATHS["csv_path"])
        except:
            df = None
    
   
    history_items = []
    history_buttons = []

    if temp_assignments:

        rounds = set()
        for key in temp_assignments.keys():
            if key.startswith('round_'):
                round_num = key.split('_')[1]
                rounds.add(int(round_num))
        
        for round_num in sorted(rounds):
            history_buttons.append(
                html.Button(f"Round {round_num} Adjustments", 
                           id={"type": "history-round", "index": round_num},
                           style={
                               "backgroundColor": "#3498db",
                               "color": "white",
                               "border": "none",
                               "padding": "8px 16px",
                               "borderRadius": "4px",
                               "margin": "5px",
                               "cursor": "pointer",
                               "fontSize": "0.9rem"
                           })
            )
    
    for entry in changes:
        change = entry.get("change", {})
        idx = change.get("document_index")
        if idx is None:
            continue
        original_group = change.get("from_group", "Unknown")
        new_group = change.get("to_group", "Unknown")
        
        doc_preview = change.get("doc_preview", "...")
        if doc_preview == "..." and df is not None and idx < len(df):
            doc_text = str(df.iloc[idx, 1])
            doc_preview = doc_text[:50] + "..." if len(doc_text) > 50 else doc_text
        
        color_from = get_group_color(original_group)
        color_to = get_group_color(new_group)
        


        display_original_group = original_group
        display_new_group = new_group
        

        try:
            if hasattr(globals(), 'current_group_order') and current_group_order:

                if "Exclude" in current_group_order and current_group_order["Exclude"]:
                    if display_new_group == "Other":
                        display_new_group = "Exclude"
                    if display_original_group == "Other":
                        display_original_group = "Exclude"
        except Exception as e:
            pass

        if original_group != new_group and original_group != "Unknown":
            change_text = f"Doc {idx+1}: {display_original_group} -> {display_new_group}"
            change_color = "#27ae60"    
        elif original_group == "Unknown":
            change_text = f"Doc {idx+1}: -> {new_group} (moved to {new_group})"
            change_color = "#e67e22" 
        else:
            continue
        
        history_items.append(
            html.Div([
                html.Div([
                    html.Span(f"Doc {idx+1}", style={  
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                        "marginRight": "10px"
                    }),
                    html.Span("->", style={"margin": "0 5px", "color": "#95a5a6"}),
                ], style={"marginBottom": "5px"}),
                html.Div([
                    html.Span(original_group, style={
                        "backgroundColor": color_from,
                        "color": "white",
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontSize": "0.85rem",
                        "marginRight": "5px"
                    }),
                    html.Span("->", style={"margin": "0 5px", "color": "#95a5a6"}),
                    html.Span(new_group, style={
                        "backgroundColor": color_to,
                        "color": "white",
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontSize": "0.85rem"
                    })
                ], style={"marginBottom": "8px"}),
                html.Div(change_text, style={
                    "fontSize": "0.8rem",
                    "color": change_color,
                    "fontWeight": "bold",
                    "marginBottom": "5px"
                }),
                html.Div(doc_preview, style={
                    "fontSize": "0.8rem",
                    "color": "#7f8c8d",
                    "fontStyle": "italic",
                    "paddingLeft": "10px",
                    "borderLeft": "2px solid #ecf0f1"
                })
            ], style={
                "padding": "10px",
                "marginBottom": "10px",
                "backgroundColor": "white",
                "borderRadius": "6px",
                "border": "1px solid #e9ecef"
            })
        )
    
    return [
        html.Div(f"Total adjustments: {len(temp_assignments)}", style={
            "fontWeight": "bold",
            "color": "#2c3e50",
            "marginBottom": "15px",
            "textAlign": "center",
            "fontSize": "1rem"
        }),
        html.Div(history_items, style={"overflowY": "auto"})
    ], history_buttons


@app.callback(
    [Output("finetune-train-btn", "children", allow_duplicate=True),
     Output("finetune-train-btn", "style", allow_duplicate=True),
     Output("finetune-training-status", "children", allow_duplicate=True),
     Output("finetune-selected-group", "data", allow_duplicate=True),
     Output("finetune-selected-keyword", "data", allow_duplicate=True), 
     Output("finetune-temp-assignments", "data", allow_duplicate=True),
     Output("finetune-highlight-core", "data", allow_duplicate=True),  
     Output("training-figures", "data", allow_duplicate=True),
     Output("finetune-train-btn", "disabled", allow_duplicate=True)],  
    Input("finetune-train-btn", "n_clicks"),
    [State("finetune-temp-assignments", "data"),
     State("group-order", "data"),
     State("training-figures", "data"),
     State("finetune-selected-group", "data")],
    prevent_initial_call=True
)
def run_finetune_training(n_clicks, temp_assignments, group_order, current_training_figures, current_selected_group):
    if not n_clicks:
        raise PreventUpdate

    # Refinement阶段：不再跑BM25/gap1/gap2。
    # 仅使用用户当前调整后的group_docs，再训练一次并更新模型与图。
    try:
        if not group_order:
            raise PreventUpdate

        success_style = {
            "backgroundColor": "#27ae60",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(39, 174, 96, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }

        global training_in_progress, FINETUNE_LAST_LOSS, df
        training_in_progress = True
        FINETUNE_LAST_LOSS = None
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])

        snapshot_for_training = get_latest_snapshot_for_training()
        if not snapshot_for_training or not snapshot_for_training.get("group_docs"):
            training_in_progress = False
            return "Run Finetune Training", success_style, "No existing group_docs snapshot. Please run initial training first.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False
        
        fig_before, fig_after, _gap_warning_text = run_training(
            skip_gap_filters=True,
            only_first_stage_training=True
        )
        if fig_before is None or fig_after is None:
            training_in_progress = False
            return "Run Finetune Training", success_style, "Refinement training failed: no figures produced.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False

        fig_before_dict = {
            "data": [trace.to_plotly_json() for trace in fig_before.data],
            "layout": fig_before.layout.to_plotly_json()
        }
        fig_after_dict = {
            "data": [trace.to_plotly_json() for trace in fig_after.data],
            "layout": fig_after.layout.to_plotly_json()
        }
        updated_figures = {"before": fig_before_dict, "after": fig_after_dict}

        training_in_progress = False
        return "Run Finetune Training", success_style, f"{'loss=' + format(FINETUNE_LAST_LOSS, '.4f') if FINETUNE_LAST_LOSS is not None else ''}", None, None, {}, [], updated_figures, False

        matched_dict_adjusted = {g: list(idxs) for g, idxs in snapshot_for_training.get("group_docs", {}).items()}
        for grp_name in group_order.keys():
            if grp_name not in matched_dict_adjusted:
                matched_dict_adjusted[grp_name] = []

        if temp_assignments:
            for idx_str, target_group in temp_assignments.items():
                if idx_str.endswith("_original"):
                    continue
                try:
                    idx = int(idx_str)
                except Exception:
                    continue
                for grp_name in matched_dict_adjusted.keys():
                    if idx in matched_dict_adjusted[grp_name]:
                        matched_dict_adjusted[grp_name].remove(idx)
                matched_dict_adjusted.setdefault(target_group, []).append(idx)

        matched_dict_adjusted = dedupe_group_docs_by_priority(matched_dict_adjusted, group_order)

        model_path = FILE_PATHS["triplet_trained_encoder"]
        device_local = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location="cpu")
        _assert_locked_checkpoint(state_dict)
        encoder = SentenceEncoder(device="cpu")
        encoder.load_state_dict(state_dict, strict=True)
        encoder = encoder.to(device_local)
        tokenizer = BertTokenizer.from_pretrained(LOCKED_BERT_NAME)
        texts = df.iloc[:, 1].fillna("").astype(str).tolist()

        def build_triplets(current_matched):
            triplets = []
            pos_groups = {g: idxs for g, idxs in current_matched.items() if g != "Exclude" and len(idxs) >= 2}
            for g, idxs in pos_groups.items():
                neg_pool = []
                for og, oidx in current_matched.items():
                    if og != g:
                        neg_pool.extend(oidx)
                if not neg_pool:
                    continue
                for a in idxs:
                    pos_candidates = [x for x in idxs if x != a]
                    if not pos_candidates:
                        continue
                    p = random.choice(pos_candidates)
                    n = random.choice(neg_pool)
                    if n != a and n != p:
                        triplets.append((a, p, n))
            random.shuffle(triplets)
            return triplets

        triplets = build_triplets(matched_dict_adjusted)
        if not triplets:
            training_in_progress = False
            return "Run Finetune Training", success_style, "No valid triplets after refinement.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False

        lr = get_config("triplet_lr", 1e-5)
        epochs = max(1, int(get_config("triplet_epochs", 10)))
        batch_size = get_config("triplet_batch_size", 16)
        margin = get_config("triplet_margin", 1.2)

        encoder.train()
        opt = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=lr)
        for _ in range(epochs):
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                a_idx = [t[0] for t in batch]
                p_idx = [t[1] for t in batch]
                n_idx = [t[2] for t in batch]
                toks_a = tokenizer([texts[j] for j in a_idx], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks_p = tokenizer([texts[j] for j in p_idx], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks_n = tokenizer([texts[j] for j in n_idx], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks_a = {k: v.to(device_local) for k, v in toks_a.items()}
                toks_p = {k: v.to(device_local) for k, v in toks_p.items()}
                toks_n = {k: v.to(device_local) for k, v in toks_n.items()}
                za = encoder.encode_tokens(toks_a)
                zp = encoder.encode_tokens(toks_p)
                zn = encoder.encode_tokens(toks_n)
                loss = nn.TripletMarginLoss(margin=margin, p=2)(za, zp, zn)
                FINETUNE_LAST_LOSS = float(loss.detach().item()) if loss is not None else FINETUNE_LAST_LOSS
                opt.zero_grad()
                loss.backward()
                opt.step()

        encoder.eval()
        with torch.no_grad():
            all_embeds = []
            enc_bs = get_config("encoding_batch_size", 64)
            for i in range(0, len(texts), enc_bs):
                toks = tokenizer(texts[i:i + enc_bs], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k: v.to(device_local) for k, v in toks.items()}
                all_embeds.append(encoder.encode_tokens(toks).cpu())
            Z_after = torch.cat(all_embeds, dim=0).numpy()

        np.save(FILE_PATHS["embeddings_trained"], Z_after)
        write_state_dict_atomic(FILE_PATHS["triplet_trained_encoder"], encoder.state_dict())

        init_coords = None
        try:
            if current_training_figures and current_training_figures.get("after"):
                base_trace = current_training_figures["after"]["data"][0]
                xs = base_trace.get("x")
                ys = base_trace.get("y")
                if xs is not None and ys is not None and len(xs) == len(Z_after):
                    init_coords = np.column_stack([xs, ys])
        except Exception:
            init_coords = None

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init=init_coords if init_coords is not None else "pca", verbose=0)
        projected_2d_after = tsne.fit_transform(Z_after)
        group_centers = {}
        for grp_name, indices in matched_dict_adjusted.items():
            valid_indices = [x for x in indices if 0 <= x < len(projected_2d_after)]
            if valid_indices:
                group_centers[grp_name] = projected_2d_after[valid_indices].mean(axis=0)

        fig_after = go.Figure()
        fig_after.add_trace(go.Scatter(
            x=projected_2d_after[:, 0],
            y=projected_2d_after[:, 1],
            mode='markers',
            name="All Documents",
            marker=dict(color=PLOT_STYLES["background"]["color"], size=PLOT_STYLES["background"]["size"], opacity=PLOT_STYLES["background"]["opacity"]),
            text=[f"Doc {i+1}" for i in range(len(projected_2d_after))],
            customdata=[[i] for i in range(len(projected_2d_after))],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        for grp_name, c2d in group_centers.items():
            fig_after.add_trace(go.Scatter(
                x=[c2d[0]], y=[c2d[1]], mode='markers+text', name=f'Center: {grp_name}',
                marker=dict(color=get_group_color(grp_name), size=PLOT_STYLES["center"]["size"], symbol=PLOT_STYLES["center"]["symbol"]),
                text=[grp_name], textposition="top center"
            ))
        fig_after.update_layout(title="2D Projection After Refinement Retraining", hovermode='closest')

        fig_before_dict = current_training_figures.get("before") if current_training_figures else None
        fig_after_dict = {
            "data": [trace.to_plotly_json() for trace in fig_after.data],
            "layout": fig_after.layout.to_plotly_json()
        }
        updated_figures = {"before": fig_before_dict, "after": fig_after_dict}

        training_in_progress = False
        return "Run Finetune Training", success_style, f"{'loss=' + format(FINETUNE_LAST_LOSS, '.4f') if FINETUNE_LAST_LOSS is not None else ''}", None, None, {}, [], updated_figures, False

    except Exception:
        traceback.print_exc()
        training_in_progress = False
        return "Run Finetune Training", {
            "backgroundColor": "#e74c3c",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(231, 76, 60, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }, "Finetune training error.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False
    



@app.callback(
    [Output("finetune-train-btn", "children", allow_duplicate=True),
     Output("finetune-train-btn", "style", allow_duplicate=True),
     Output("finetune-training-status", "children", allow_duplicate=True),
     Output("finetune-train-btn", "disabled", allow_duplicate=True)],
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True
)
def update_training_status(n_intervals):
    global training_in_progress, FINETUNE_LAST_LOSS

    loss_text = ""
    if FINETUNE_LAST_LOSS is not None:
        try:
            loss_text = f"loss={FINETUNE_LAST_LOSS:.4f}"
        except Exception:
            loss_text = f"loss={FINETUNE_LAST_LOSS}"
    
    if training_in_progress:
        running_style = {
            "backgroundColor": "#f39c12",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "not-allowed",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(243, 156, 18, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }
        return "Running...", running_style, loss_text, True
    else:
        normal_style = {
            "backgroundColor": "#9b59b6",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(155, 89, 182, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }
        return "Run Finetune Training", normal_style, (loss_text if loss_text else ""), False



@app.callback(
    Output("train-btn", "style", allow_duplicate=True),
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_train_button_visibility(display_mode):
    if display_mode == "keywords":
        return {
            "backgroundColor": "#e74c3c",
            "color": "white",
            "border": "none",
            "padding": "10px 20px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "minWidth": "120px",
            "flexShrink": "0"
        }
    else:
        return {"display": "none"}



@app.callback(
    Output("switch-finetune-btn", "style", allow_duplicate=True),
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_finetune_button_visibility(display_mode):
    if display_mode == "training":
        return {
            "backgroundColor": "#8e44ad",
            "color": "white",
            "border": "none",
            "padding": "10px 20px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "minWidth": "180px",
            "flexShrink": "0"
        }
    else:
        return {"display": "none"}


@app.callback(
    Output("user-name-store", "data"),
    Input("user-name", "value"),
    prevent_initial_call=False
)
def update_user_name(value):
    global CURRENT_USER_NAME
    name = (value or "Yan").strip()
    if not name:
        name = "Yan"
    CURRENT_USER_NAME = name
    return name
@app.callback(
    [Output("group-order", "data", allow_duplicate=True),
     Output("new-keyword-input", "value", allow_duplicate=True),
     Output("status-output", "children", allow_duplicate=True)],
    Input("add-keyword-btn", "n_clicks"),
    [State("new-keyword-input", "value"),
     State("group-order", "data"),
     State("selected-group", "data")],
    prevent_initial_call=True
)
def add_custom_keyword(n_clicks, keyword_value, group_order, selected_group):
    if not n_clicks or not keyword_value or not selected_group:
        raise PreventUpdate
    
    if not group_order:
        group_order = {}
    
    if selected_group not in group_order:
        group_order[selected_group] = []
    

    keywords = [kw.strip() for kw in keyword_value.split(',') if kw.strip()]
    
    added_keywords = []
    already_exists = []
    
    for keyword in keywords:
        if keyword not in group_order[selected_group]:
            group_order[selected_group].append(keyword)
            added_keywords.append(keyword)
        else:
            already_exists.append(keyword)
    
   
    if added_keywords and already_exists:
        message = f"Added: {', '.join(added_keywords)} | Already exists: {', '.join(already_exists)}"
    elif added_keywords:
        message = f"Added {len(added_keywords)} keywords: {', '.join(added_keywords)}"
    else:
        message = f"All keywords already exist: {', '.join(already_exists)}"
        return dash.no_update, "", message
    
    update_live_keywords_snapshot(group_order)
    return group_order, "", message


if __name__ == "__main__":
    os.environ['FLASK_ENV'] = 'development'
    app.run(
            debug=True,
            port=47983,
            host='127.0.0.1',
            use_reloader=False,
            threaded=True
        )
