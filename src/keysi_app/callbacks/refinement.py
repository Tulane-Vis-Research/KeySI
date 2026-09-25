"""Callbacks for document refinement and refinement training."""

from .. import core
from ..core import (
    ALL,
    BertTokenizer,
    FILE_PATHS,
    FINETUNE_LAST_LOSS,
    Input,
    LOCKED_BERT_NAME,
    Output,
    PLOT_STYLES,
    PreventUpdate,
    RANDOM_SEED,
    SentenceEncoder,
    State,
    TSNE,
    _assert_locked_checkpoint,
    contains_keyword_word_boundary,
    dash,
    df,
    get_config,
    get_document_tsne,
    get_group_color,
    get_user_data_path,
    go,
    html,
    json,
    json_module,
    keywords,
    nn,
    np,
    os,
    pd,
    perplexity,
    random,
    torch,
    traceback,
    training_in_progress,
    tsne,
)
from ..training import (
    dedupe_group_docs_by_priority,
    filter_keyword_matches_in_group,
    get_latest_refinement_snapshot,
    get_latest_snapshot_for_training,
    get_latest_training_snapshot,
    load_keysi_user_data,
    record_user_data,
    update_live_keywords_snapshot,
    write_json_atomic,
    write_state_dict_atomic,
)
from ..ui import app

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
        if not os.path.exists(model_path):
            training_in_progress = False
            return "Run Finetune Training", success_style, "No trained encoder checkpoint found. Please run initial training first.", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, False

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

        n_samples = len(Z_after)
        perplexity = min(30, max(5, n_samples // 3))
        perplexity = min(perplexity, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity, init=init_coords if init_coords is not None else "pca", verbose=0)
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
    name = (value or "Yan").strip()
    if not name:
        name = "Yan"
    core.CURRENT_USER_NAME = name
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
