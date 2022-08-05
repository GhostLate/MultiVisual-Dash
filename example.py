import json
import random
import time

import numpy as np

from client import WebSocketClient
from dash_visualizer import DashVisualizer
from utils import NumpyEncoder

if __name__ == "__main__":
    rand = random.Random()

    plot_name = "name"
    address = "localhost"
    ws_port = 4002
    websocket_url = f"ws://{address}:{ws_port}"

    visualizer = DashVisualizer(plot_name, address, ws_port, 8002, websocket_url)
    viz = WebSocketClient(websocket_url)

    for i in range(20):
        plot_data = {
            'command_type': 'add2plot',
            'plot_name': plot_name,
            'lines': [{
                'line_name': "1",
                'data': {
                    'x': [i * rand.uniform(-1., 1.)],
                    'y': np.array([2 * i * rand.uniform(-1., 1.)]),
                    'z': np.array([0.1 * i * rand.uniform(-1., 1.)]),
                    'xr': [rand.uniform(-1., 1.)],
                    'yr': np.array([rand.uniform(-1., 1.)]),
                    'legend': "ped"
                }
            }]
        }
        viz.send(json.dumps(plot_data, cls=NumpyEncoder))
        time.sleep(0.2)

    plot_names = [plot_name, '11', '22', '33']
    line_names = ['q', 'd', 'f', 'v']
    for plot in plot_names:
        for line_name in line_names:
            data = np.zeros((4, 30))
            for i in range(data.shape[1]):
                data[0, i] = np.array([i * rand.uniform(-1., 1.)])
                data[1, i] = np.array([2 * i * rand.uniform(-1., 1.)])
                data[2, i] = np.array([rand.uniform(-1., 1.)])
                data[3, i] = np.array([rand.uniform(-1., 1.)])

            plot_data = {
                'command_type': 'add_plot',
                'plot_name': plot,
                'lines': [{
                    'line_name': line_name,
                    'data': {
                        'x': data[0, :],
                        'y': data[1, :],
                        'xr': data[2, :],
                        'yr': data[3, :],
                        'legend': "car"
                    }
                }]
            }
            viz.send(json.dumps(plot_data, cls=NumpyEncoder))
            time.sleep(0.2)
