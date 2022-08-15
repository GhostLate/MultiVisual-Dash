import random
import time

import numpy as np

from dash_visualizer import DashVisualizer
from websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4002
    websocket_url = f"ws://{address}:{ws_port}"

    visualizer_server = DashVisualizer('Trajectory', address, ws_port, 8003, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    for i in range(40):
        plot_data = {
            'command_type': 'add2plot',
            'plot_name': "00",
            'scatters': [{
                'mode': 'markers',
                'name': "1",
                'x': [random.uniform(-1., 1.)],
                'y': [random.uniform(-1., 1.)],
                'z': [random.uniform(-1., 1.)],
                'desc': "ped"
            }]
        }
        visualizer_client.send(plot_data)
        time.sleep(0.1)

    for plot in ["00", '11', '22', '33']:
        plot_data = {
            'mode': 'lines+markers',
            'command_type': 'add_plot',
            'plot_name': plot,
            'scatters': []
        }
        for line_name in ['w', 'a', 's', 'd']:
            scatter = {
                    'mode': 'lines+markers',
                    'name': line_name,
                    'x': np.random.uniform(-1., 1., size=20),
                    'y': np.random.uniform(-1., 1., size=20),
                    'desc': "car"
            }
            plot_data['scatters'].append(scatter)
        visualizer_client.send(plot_data)
        time.sleep(0.2)
