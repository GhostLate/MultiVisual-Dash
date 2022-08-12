import json
import multiprocessing
import os

from dash.dependencies import Input, Output
from dash_extensions.enrich import DashProxy

from .layout import init_layout
from .utils import CustomFigure


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
        self.figure = CustomFigure(self.plots_data)

    def run(self):
        self.app.layout = init_layout(self)
        self.app.clientside_callback(self.update_graph_func,
                                     Output('plots_data_store', 'data'),
                                     Input("ws", "message"), prevent_initial_call=True)

        self.app.callback(Output('name-dropdown', 'options'),
                          Output('name-dropdown', 'value'),
                          Input('plots_data_store', 'data'))(self.update_data)

        self.app.callback(Output('live-update-graph', 'figure'),
                          Input('name-dropdown', 'value'))(self.change_cur_plot)

        self.app.run_server(host=self.host, port=self.port, dev_tools_silence_routes_logging=True, debug=False)

    def update_data(self, msg):
        if msg:
            msg_data = json.loads(msg['msg']['data'])
            self.update_plots_data(msg_data)
            if 'save_dir' in msg_data:
                save_p = multiprocessing.Process(target=save_plot_as_img, kwargs={
                    'plot_data': self.plots_data[msg_data['plot_name']],
                    'plot_name': msg_data['plot_name'],
                    'save_dir': msg_data['save_dir']
                })
                save_p.start()

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
        if 'title' not in msg_data:
            plot_data['title'] = self.title
        for scatter in msg_data['scatters']:
            plot_data['scatters'].setdefault(scatter['name'], {})
            plot_scatter = plot_data['scatters'][scatter['name']]

            for key, value in scatter.items():
                if key in plot_scatter and msg_data['command_type'] == 'add2plot' \
                        and isinstance(value, list) and isinstance(plot_scatter[key], list):
                    plot_scatter[key].extend(value)
                else:
                    plot_scatter[key] = value

            if 'type' in plot_scatter:
                plot_data['scatter_types'].add(plot_scatter['type'])

            if all(key in plot_scatter for key in ['x', 'y', 'z']):
                plot_data['type'] = '3D'
            elif all(key in plot_scatter for key in ['x', 'y']):
                plot_data['type'] = '2D'

    def change_cur_plot(self, value):
        self.cur_plot = value
        return self.figure.update(self.cur_plot)


def save_plot_as_img(plot_data: dict, plot_name, save_dir: str,
                     plot_scale: float = 1.0, img_w: int = 3840, img_h: int = 2160, img_format: str = 'svg'):
    if any(key == plot_data['type'] for key in ['2D', '3D']):
        try:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plots_data = {plot_name: plot_data}
            figure = CustomFigure(plots_data).update(plot_name)
            img_path = f'{save_dir}/{plot_name}.{img_format}'
            figure.write_image(img_path, scale=plot_scale, width=img_w, height=img_h)
            print(f'Saved to {img_path}')
            return True
        except Exception as ex:
            print(ex)
