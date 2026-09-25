"""Training, persistence, retrieval, and evaluation routines for KeySI."""

from .core import (
    BM25Okapi,
    BertTokenizer,
    Counter,
    FILE_PATHS,
    LOCKED_BERT_NAME,
    OUTPUT_DIR,
    PLOT_STYLES,
    PreventUpdate,
    RANDOM_SEED,
    SentenceEncoder,
    SnowballStemmer,
    TSNE,
    _BM25_CACHE,
    _KEYWORD_MATCH_CACHE,
    _USER_DATA_CACHE,
    _assert_locked_checkpoint,
    contains_keyword_word_boundary,
    df,
    get_config,
    get_group_color,
    get_safe_user_name,
    get_user_data_path,
    get_user_model_dir,
    go,
    json,
    keyword_embeddings,
    keywords,
    kw_model,
    labels,
    logger,
    nn,
    np,
    os,
    pd,
    plt,
    random,
    re,
    reset_user_data_on_start,
    save_user_data_to_user_dir,
    silhouette_score,
    torch,
    traceback,
    word_tokenize,
)

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
        logger.warning("Failed to load KeySI user data.", exc_info=True)
    return {}

def write_json_atomic(path, data):
    try:
        base_dir = os.path.dirname(path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        logger.warning("Failed to write JSON atomically to %s.", path, exc_info=True)
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
        logger.warning("Failed to write state dict atomically to %s.", path, exc_info=True)
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
        logger.warning("Failed to update live keyword snapshot.", exc_info=True)

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
            random_state=RANDOM_SEED,
            max_iter=get_config("tsne_max_iter"),
            verbose=0
        ).fit_transform(Z_raw)
        
        X2_trained = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=RANDOM_SEED,
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
        random_state=RANDOM_SEED,
        max_iter=get_config("tsne_max_iter"),
        verbose=0
    )
    tsne_after = TSNE(
        n_components=2,
        perplexity=perplexity_after,
        random_state=RANDOM_SEED,
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
        random_state=RANDOM_SEED,
        max_iter=get_config("tsne_max_iter"),
        verbose=0
    )
    tsne_after = TSNE(
        n_components=2,
        perplexity=perplexity_after,
        random_state=RANDOM_SEED,
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



