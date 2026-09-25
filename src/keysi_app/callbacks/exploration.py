"""Callbacks for keyword grouping, exploration, and initial training."""

from ..core import (
    ALL,
    BertTokenizer,
    FILE_PATHS,
    GLOBAL_KEYWORDS,
    GLOBAL_OUTPUT_DICT,
    Input,
    LOCKED_BERT_NAME,
    Output,
    PLOT_STYLES,
    PreventUpdate,
    RANDOM_SEED,
    SentenceEncoder,
    State,
    TSNE,
    _GLOBAL_DOCUMENT_EMBEDDINGS,
    _GLOBAL_DOCUMENT_EMBEDDINGS_READY,
    all_articles_text,
    contains_keyword_word_boundary,
    dash,
    df,
    get_document_tsne,
    get_group_color,
    get_valid_doc2d_index_map,
    html,
    json,
    keyword_embeddings,
    keywords,
    np,
    os,
    pd,
    perplexity,
    reduced_embeddings,
    torch,
    traceback,
    truncate_text_for_model,
    tsne,
)
from ..training import (
    _ARTICLES_CACHE,
    _DOCUMENTS_2D_CACHE,
    _KEYWORD_TSNE_CACHE,
    clear_caches,
    dedupe_group_docs_by_priority,
    extract_top_keywords,
    filter_keyword_matches_in_group,
    get_group_doc_indices_cached,
    get_keysi_user_data_mtime,
    get_keyword_doc_indices_cached,
    get_latest_training_snapshot,
    record_user_data,
    resolve_group_for_keyword,
    run_training,
    update_live_keywords_snapshot,
)
from ..ui import app

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
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED, verbose=0)
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
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED, n_jobs=-1, verbose=0)
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
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED, n_jobs=-1, verbose=0)
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
