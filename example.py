import random
import time

import numpy as np

from .dash_visualizer import DashVisualizer
from .websocket.client import WebSocketClient

if __name__ == "__main__":
    rand = random.Random()

    plot_name = "name"
    address = "localhost"
    ws_port = 4002
    websocket_url = f"ws://{address}:{ws_port}"

    visualizer_server = DashVisualizer('Trajectory', address, ws_port, 8003, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    for i in range(20):
        plot_data = {
            'command_type': 'add2plot',
            'plot_name': plot_name,
            'scatters': [{
                'mode': 'markers',
                'name': "1",
                'x': [i * rand.uniform(-1., 1.)],
                'y': np.array([2 * i * rand.uniform(-1., 1.)]),
                'z': np.array([0.1 * i * rand.uniform(-1., 1.)]),
                'desc': "ped"
            }]
        }
        visualizer_client.send(plot_data)
        time.sleep(0.2)

    plot_names = [plot_name, '11', '22', '33']
    line_names = ['q', 'd', 'f', 'v']
    for plot in plot_names:
        plot_data = {
            'mode': 'lines+markers',
            'command_type': 'add_plot',
            'plot_name': plot,
            'scatters': []
        }
        for line_name in line_names:
            data = np.zeros((4, 30))
            for i in range(data.shape[1]):
                data[0, i] = np.array([i * rand.uniform(-1., 1.)])
                data[1, i] = np.array([2 * i * rand.uniform(-1., 1.)])
            scatter = {
                    'mode': 'lines+markers',
                    'name': line_name,
                    'x': data[0, :],
                    'y': data[1, :],
                    'desc': "car"
            }
            plot_data['scatters'].append(scatter)
        visualizer_client.send(plot_data)
        time.sleep(0.2)
