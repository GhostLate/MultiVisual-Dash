import random
import time

import numpy as np

from multi_visual_dash.dash_visualizer import DashVisualizer
from multi_visual_dash.dash_viz.data import DashMessage, ScatterData
from multi_visual_dash.websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4000
    websocket_url = f"ws://{address}:{ws_port}"

    visualizer_server = DashVisualizer('Trajectory', address, ws_port, 8000, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    for i in range(40):
        viz_massage = DashMessage('add2plot', "00")
        scatter = ScatterData(
            "1",
            'markers',
            [random.uniform(-1., 1.), ],
            [random.uniform(-1., 1.), ],
            [random.uniform(-1., 1.), ])
        scatter.desc = "ped"
        viz_massage.scatters.append(scatter)
        visualizer_client.send(dict(viz_massage))
        time.sleep(0.1)

    for plot in ["00", '11', '22', '33']:
        viz_massage = DashMessage('add_plot', plot, True)
        for line_name in ['w', 'a', 's', 'd']:
            scatter = ScatterData(
                line_name,
                'lines+markers',
                np.random.uniform(-1., 1., size=20),
                np.random.uniform(-1., 1., size=20))
            scatter.desc = "ped"
            viz_massage.scatters.append(scatter)
        visualizer_client.send(dict(viz_massage))
        time.sleep(0.2)
