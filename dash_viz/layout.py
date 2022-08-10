from dash import dcc
from dash import html
from dash_extensions import WebSocket


def init_layout(self) -> html.Div:
    return html.Div([
        html.Div(
            [
                html.Span(children='Graph: ',
                          style={
                              'display': 'inline-block',
                              'margin': 15,
                              'fontSize': '20px',
                              'padding': '0px',
                              'verticalAlign': 'middle',
                              'color': 'white'
                          }),
                dcc.Dropdown(
                    id='name-dropdown',
                    clearable=False,
                    style={
                        'fontSize': '20px',
                        'margin': 5,
                        'width': '250px',
                        'verticalAlign': 'middle',
                    })
            ],
            className="row",
            style={
                'zIndex': '998',
                'position': 'absolute',
                'padding': '0px',
                'display': 'flex'
            }
        ),
        dcc.Loading([dcc.Graph(
            id='live-update-graph',
            style={
                'zIndex': '999',
                'height': '100%',
                'width': '100%',
                'padding': '0px',
            }
        )], type="cube", parent_style={
            'height': '100%',
            'width': '100%',
            'padding': '0px',
            }),
        dcc.Store(id='plots_data_store'),
        WebSocket(id="ws", url=self.websocket_url),
    ], style={
            'height': '100%',
            'width': '100%',
            'background': 'black',
            'position': 'absolute',
            'top': '0px',
            'left': '0px',
            'padding': '0px',
        }
    )
