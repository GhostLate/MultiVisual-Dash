import json
import multiprocessing

from dash.dependencies import Input, Output
from dash_extensions.enrich import DashProxy

from dash_viz.layout import init_layout
from dash_viz.utils import UpdateGraph


class DashApp(multiprocessing.Process):
    dropdown_options: list
    cur_plot: str | None
    plots_data: dict

    def __init__(self, title: str, host: str, port: int, websocket_url: str):
        multiprocessing.Process.__init__(self)
        self.title = title
        self.port = port
        self.host = host
        self.websocket_url = websocket_url
        self.plots_data = dict()
        self.dropdown_options = list()
        self.cur_plot = None

        self.app = DashProxy()
        self.update_graph_func = """function(msg) {
                if(!msg) {return {};} return {msg}};"""

    def run(self):
        self.app.layout = init_layout(self)
        self.app.clientside_callback(self.update_graph_func,
                                     Output('plots_data_store', 'data'),
                                     Input("ws", "message"), prevent_initial_call=True)

        self.app.callback(Output('name-dropdown', 'options'),
                          Output('name-dropdown', 'value'),
                          Input('plots_data_store', 'data'))(self.update_data)

        self.app.callback(Output('live-update-graph', 'figure'),
                          Input('name-dropdown', 'value'),
                          Input('name-dropdown', 'options'))(self.change_cur_plot)

        self.app.run_server(host=self.host, port=self.port, dev_tools_silence_routes_logging=True, debug=False)

    def update_data(self, msg):
        if msg:
            msg_data = json.loads(msg['msg']['data'])
            self.update_plots_data(msg_data)
        self.dropdown_options = []
        if self.plots_data:
            self.dropdown_options = [{'label': f"{plt_name}_{plt_data['type']}", 'value': plt_name}
                                     for plt_name, plt_data in self.plots_data.items()]

            if self.cur_plot is None:
                self.cur_plot = self.dropdown_options[0]['value']
        return self.dropdown_options, self.cur_plot

    def update_plots_data(self, msg_data):
        self.plots_data.setdefault(msg_data['plot_name'], {})
        plot_data = self.plots_data[msg_data['plot_name']]

        plot_data.setdefault('type', 'None')
        plot_data.setdefault('scatter_types', set())
        plot_data.setdefault('scatters', {})

        for scatter in msg_data['scatters']:
            plot_data['scatters'].setdefault(scatter['name'], {})
            plot_scatter = plot_data['scatters'][scatter['name']]

            for key, value in scatter.items():
                if msg_data['command_type'] == 'add2plot' and isinstance(value, list):
                    plot_scatter.setdefault(key, [])
                    plot_scatter[key].extend(value)
                else:
                    plot_scatter[key] = value

            if 'type' in plot_scatter:
                plot_data['scatter_types'].add(plot_scatter['type'])

            if all(key in plot_scatter for key in ['x', 'y', 'z']):
                plot_data['type'] = '3D'
            elif all(key in plot_scatter for key in ['x', 'y']):
                plot_data['type'] = '2D'

    def change_cur_plot(self, value, _):
        self.cur_plot = value
        return UpdateGraph(self.title, self.plots_data, self.cur_plot).fig
