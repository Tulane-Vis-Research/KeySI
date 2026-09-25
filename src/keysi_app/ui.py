"""Dash application instance and page layout for KeySI."""

from .core import dash, dcc, html, keywords

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
            interval=5000, 
            n_intervals=0
        )
    ])

app.layout = create_layout()
