import json
import multiprocessing
import os
import time

import numpy as np
from dash.dependencies import Input, Output
from dash_extensions.enrich import DashProxy

from debug.utils import timing
from multi_visual_dash.dash_viz.custom_figure import CustomFigure
from multi_visual_dash.dash_viz.layout import init_layout
from multi_visual_dash.websocket.utils import decompress_message


class DashApp(multiprocessing.Process):
    def __init__(self, title: str, host: str, port: int, websocket_url: str, use_loader_widget=False):
        multiprocessing.Process.__init__(self)
        self.title = title
        self.port = port
        self.host = host
        self.websocket_url = websocket_url
        self.plots_data = dict()
        self.dropdown_options = list()
        self.cur_plot = None
        self.use_loader_widget = use_loader_widget
        self.app = DashProxy()
        self.figure = CustomFigure(title)
        self.start()

    def run(self):
        self.app.layout = init_layout(self.websocket_url, self.use_loader_widget)

        self.app.callback(Output('name-dropdown', 'options'),
                          Output('name-dropdown', 'value'),
                          Output("ws", "send"),
                          Input("ws", "message"))(self.update_data)

        self.app.callback(Output('live-update-graph', 'figure'),
                          Input('name-dropdown', 'value'))(self.change_cur_plot)

        self.app.run_server(host=self.host, port=self.port, dev_tools_silence_routes_logging=True, debug=False)

    @timing
    def update_data(self, msg):
        if msg:
            msg_data = decompress_message(msg['data'])
            self.update_plots_data(msg_data)
            if 'save_dir' in msg_data:
                multiprocessing.Process(target=save_plot_as_img, kwargs={
                    'plot_data': self.plots_data[msg_data['plot_name']],
                    'plot_name': msg_data['plot_name'],
                    'save_dir': msg_data['save_dir']
                }).start()

        self.dropdown_options = []
        if self.plots_data:
            self.dropdown_options = [{'label': f"{plt_name}_{self.plots_data[plt_name]['type']}", 'value': plt_name}
                                     for plt_name in sorted(list(self.plots_data.copy().keys()))]
            if self.cur_plot is None:
                self.cur_plot = self.dropdown_options[0]['value']
        return self.dropdown_options, self.cur_plot, json.dumps({'ws_status': 'ready', 'time': time.time()})

    def update_plots_data(self, msg_data: dict):
        if msg_data['command_type'] == 'new_plot' or msg_data['plot_name'] not in self.plots_data:
            self.plots_data[msg_data['plot_name']] = dict()
        plot_data = self.plots_data[msg_data['plot_name']]

        if 'title' in msg_data:
            plot_data['title'] = msg_data['title']
        else:
            plot_data['title'] = self.title

        if 'scene_camera' in msg_data:
            plot_data['scene_camera'] = msg_data['scene_camera']

        if 'scene_centric_data' in msg_data:
            plot_data['scene_centric_data'] = msg_data['scene_centric_data']

        plot_data.setdefault('type', '2D')
        plot_data.setdefault('scatters', {})
        for scatter in msg_data['scatters']:
            plot_data['scatters'].setdefault(scatter['name'], {})
            plot_scatter = plot_data['scatters'][scatter['name']]

            for key, value in scatter.items():
                if isinstance(value, list):
                    value = np.array(value)

                    if msg_data['command_type'] == 'add2plot' and key in plot_scatter:
                        plot_scatter[key] = np.append(plot_scatter[key], value)
                    else:
                        plot_scatter[key] = value
                else:
                    plot_scatter[key] = value

            if all(key in plot_scatter for key in ['x', 'y', 'z']):
                plot_data['type'] = '3D'

        if plot_data['type'] == '3D':
            for scatter_name, scatter_data in plot_data['scatters'].items():
                if 'z' not in scatter_data:
                    scatter_data['z'] = np.zeros(shape=scatter_data['x'].shape)

    @timing
    def change_cur_plot(self, value):
        self.cur_plot = value
        if self.cur_plot in self.plots_data:
            return self.figure.update(self.plots_data[self.cur_plot])
        return self.figure.figure


def save_plot_as_img(plot_data: dict, plot_name, save_dir: str,
                     plot_scale: float = 1.0, img_w: int = 3840, img_h: int = 2160, img_format: str = 'png'):
    if any(key == plot_data['type'] for key in ['2D', '3D']):
        try:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            figure = CustomFigure().update(dict(plot_name=plot_data))
            img_path = f'{save_dir}/{plot_name}.{img_format}'
            figure.write_image(img_path, scale=plot_scale, width=img_w, height=img_h)
            print(f'Saved to {img_path}')
            return True
        except Exception as ex:
            print(ex)
