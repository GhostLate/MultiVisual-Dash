from dash import dcc
from dash import html
from dash_extensions import WebSocket


def init_layout(websocket_url: str, loading_widget=False) -> html.Div:
    dcc_graph = dcc.Graph(
        id='live-update-graph',
        style={
            'zIndex': '999',
            'height': '100%',
            'width': '100%',
            'padding': '0px',
        }
    )
    if loading_widget:
        dcc_graph = dcc.Loading(
            [dcc_graph],
            type="cube",
            parent_style={
                'height': '100%',
                'width': '100%',
                'padding': '0px',
            }
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        children='Graph: ',
                        style={
                            'display': 'inline-block',
                            'margin': 17,
                            'fontSize': '25px',
                            'padding': '0px',
                            'verticalAlign': 'middle',
                            'color': 'white'
                        }
                    ),
                    dcc.Dropdown(
                        id='name-dropdown',
                        clearable=False,
                        style={
                            'fontSize': '17px',
                            'margin': 5,
                            'width': '300px',
                            'verticalAlign': 'middle',
                        }
                    ),
                ],
                className="row",
                style={
                    'zIndex': '998',
                    'position': 'absolute',
                    'padding': '0px',
                    'display': 'flex'
                }
            ),
            dcc_graph,
            WebSocket(id="ws", url=websocket_url),
        ],
        style={
            'height': '100%',
            'width': '100%',
            'background': 'black',
            'position': 'absolute',
            'top': '0px',
            'left': '0px',
            'padding': '0px',
        }
    )
