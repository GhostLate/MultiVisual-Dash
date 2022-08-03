import json
import multiprocessing

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

        self.plots_data = {plot_name: {'type': '2D', 'lines': {}}}
        self.dropdown_names = [(plt_name, self.plots_data[plt_name]['type']) for plt_name in self.plots_data.keys()]
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
                        clearable=False,
                        options=[{'label': f"{plt_name}_{plt_type}",
                                  'value': plt_name} for plt_name, plt_type in self.dropdown_names],
                        value=self.cur_plot,
                        style={
                            'fontSize': '20px',
                            'margin': 5,
                            'width': '250px',
                            'verticalAlign': 'middle',
                        },
                        persistence=True,
                        persistence_type='local')
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
                          Input('name-dropdown', 'value'),
                          Input('name-dropdown', 'options'))(self.change_cur_plot)

        self.app.run_server(host=self.host, port=self.port, dev_tools_silence_routes_logging=True, debug=False)

    def update_data(self, message):
        command = json.loads(message['msg']['data'])
        plot_name = command['plot_name']
        self.plots_data.setdefault(plot_name, {})
        self.plots_data[plot_name].setdefault('lines', {})
        self.plots_data[plot_name].setdefault('type', '2D')
        for line in command['lines']:
            line_name = line['line_name']
            plots_lines = self.plots_data[plot_name]['lines']
            plots_lines.setdefault(line_name, {})
            for key, value in line['data'].items():
                if isinstance(value, list):
                    value = np.array(value)
                    plots_lines[line_name].setdefault(key, np.array([]))
                    if command['command_type'] == 'add2plot':
                        plots_lines[line_name][key] = np.append(plots_lines[line_name][key], value)
                        continue
                plots_lines[line_name][key] = value
            if all(key in plots_lines[line_name] for key in ['x', 'y', 'z']):
                self.plots_data[plot_name]['type'] = '3D'

        self.dropdown_names = [(plt_name, self.plots_data[plt_name]['type']) for plt_name in self.plots_data.keys()]
        return [{'label': f"{plt_name}_{plt_type}", 'value': plt_name} for plt_name, plt_type in self.dropdown_names], self.cur_plot

    def change_cur_plot(self, value, _):
        self.cur_plot = value
        if self.plots_data[self.cur_plot]['type'] == '2D':
            return self.update_graph2d()
        else:
            return self.update_graph3d()

    def update_graph2d(self):
        fig = go.Figure()
        for line_name, line in self.plots_data[self.cur_plot]['lines'].items():
            if all(line[key].shape[0] > 0 for key in ['x', 'y']):
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
            uirevision=True)
        return fig

    def update_graph3d(self):
        fig = go.Figure()
        max_vals, min_vals = [], []
        for line_name, line in self.plots_data[self.cur_plot]['lines'].items():
            if 'z' not in line:
                line['z'] = np.zeros(shape=line['x'].shape, dtype=float)
            if all(line[key].shape[0] > 0 for key in ['x', 'y', 'z']):
                fig.add_scatter3d(
                    x=line['x'],
                    y=line['y'],
                    z=line['z'],
                    name=line_name,
                    mode='lines+markers')
                max_vals.append(np.max([line['x'], line['y'], line['z']]))
                min_vals.append(np.min([line['x'], line['y'], line['z']]))
        max_val, min_val = max(max_vals), min(min_vals)
        fig.update_layout(
            template="plotly_dark",
            autosize=True,
            scene=dict(
                xaxis=dict(range=[min_val, max_val]),
                yaxis=dict(range=[min_val, max_val]),
                zaxis=dict(range=[min_val, max_val]),
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Z Position',
                aspectmode='cube'
            ),
            uirevision=True)
        return fig
