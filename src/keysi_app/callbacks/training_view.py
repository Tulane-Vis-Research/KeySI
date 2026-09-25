"""Callbacks for inspecting trained document projections."""

from ..core import (
    ALL,
    BM25Okapi,
    FILE_PATHS,
    Input,
    Output,
    PreventUpdate,
    SnowballStemmer,
    State,
    contains_keyword_word_boundary,
    dash,
    dcc,
    df,
    get_group_color,
    html,
    json,
    keywords,
    os,
    pd,
    re,
    traceback,
    word_tokenize,
)
from ..training import (
    _ARTICLES_CACHE,
    dedupe_group_docs_by_priority,
    extract_top_keywords,
    filter_keyword_matches_in_group,
    get_keysi_user_data_mtime,
    get_latest_training_snapshot,
    resolve_group_for_keyword,
)
from ..ui import app

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
