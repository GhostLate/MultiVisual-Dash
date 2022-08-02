import json
import multiprocessing
import time

import numpy as np
import plotly.graph_objs as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash_extensions import WebSocket
from dash_extensions.enrich import DashProxy


class DashApp(multiprocessing.Process):
    def __init__(self, plot_name: str, host: str, port: int, websocket_url: str):
        multiprocessing.Process.__init__(self)
        self.port = port
        self.host = host
        self.websocket_url = websocket_url

        self.plots_data = {plot_name: {}}
        self.dropdown_names = list(self.plots_data.keys())
        self.cur_plot = plot_name
        self.app = DashProxy()
        self.update_graph_func = """function(msg) {
                if(!msg) {return {};} return {msg}};"""

    def init_layout(self):
        return html.Div([
            html.H4(
                'Trajectory',
                style={
                    'textAlign': 'center',
                    'color': 'white',
                    'fontSize': '30px',
                    'margin': 0,
                    'padding': '8px',
                }
            ),
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
                        options=[{'label': self.cur_plot, 'value': self.cur_plot}],
                        clearable=False,
                        value=self.cur_plot,
                        style={
                            'fontSize': '20px',
                            'margin': 5,
                            'width': '250px',
                            'verticalAlign': 'middle',
                        }
                    )
                ],
                className="row",
                style={
                    'zIndex': '998',
                    'position': 'absolute',
                    'padding': '0px',
                    'display': 'flex'
                }
            ),
            dcc.Graph(
                id='live-update-graph',
                style={
                    'zIndex': '999',
                    'height': '90%',
                    'width': '100%',
                    'padding': '0px',
                }
            ),
            dcc.Store(id='plots_data_store'),
            WebSocket(id="ws", url=self.websocket_url),
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

    def run(self):
        self.app.layout = self.init_layout()

        self.app.clientside_callback(self.update_graph_func,
                                     Output('plots_data_store', 'data'),
                                     Input("ws", "message"), prevent_initial_call=True)

        self.app.callback(Output('name-dropdown', 'options'),
                          Output('name-dropdown', 'value'),
                          Input('plots_data_store', 'data'), prevent_initial_call=True)(self.update_data)

        self.app.callback(Output('live-update-graph', 'figure'),
                          Input('name-dropdown', 'value'))(self.change_cur_plot)

        self.app.run_server(host=self.host, port=self.port, dev_tools_silence_routes_logging=True, debug=False)

    def update_data(self, message):
        command = json.loads(message['msg']['data'])
        plot_name = command['plot_name']
        self.plots_data.setdefault(plot_name, {})
        for line in command['lines']:
            line_name = line['line_name']
            self.plots_data[plot_name].setdefault(line_name, {})
            for key, value in line['data'].items():
                if isinstance(value, list):
                    value = np.array(value)
                    self.plots_data[plot_name][line_name].setdefault(key, np.array([]))
                    if command['command_type'] == 'add2plot':
                        self.plots_data[plot_name][line_name][key] = np.append(
                            self.plots_data[plot_name][line_name][key], value)
                    else:
                        self.plots_data[plot_name][line_name][key] = value
                else:
                    self.plots_data[plot_name][line_name][key] = value
        self.dropdown_names = list(self.plots_data.keys())
        return [{'label': i, 'value': i} for i in self.dropdown_names], self.cur_plot

    def change_cur_plot(self, value):
        self.cur_plot = value
        return self.update_graph()

    def update_graph(self):
        fig = go.Figure()
        for line_name, line in self.plots_data[self.cur_plot].items():
            if line['x'].shape[0] > 0 and line['y'].shape[0] > 0:
                fig.add_scatter(
                    x=line['x'],
                    y=line['y'],
                    name=line_name,
                    mode='lines+markers')
        fig.update_xaxes(title="X Position",
                         scaleanchor='y')
        fig.update_yaxes(title="Y Position",
                         constrain='domain')
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                aspectratio=dict(x=1, y=1)
            ),
        )
        fig.update_layout(uirevision=True)
        return fig
