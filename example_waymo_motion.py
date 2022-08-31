import time

from multi_visual_dash.dash_visualizer import DashVisualizer
from multi_visual_dash.dataloaders.waymo.motion.visualizer import WaymoMotionVisualizer
from multi_visual_dash.websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4000
    websocket_url = f"ws://{address}:{ws_port}"

    waymo_data_visualizer = WaymoMotionVisualizer("data/motion/", center_data=True)
    visualizer_server = DashVisualizer('Waymo Motion', address, ws_port, 8000, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    for idx, plot_data in enumerate(waymo_data_visualizer()):
        visualizer_client.send(dict(plot_data))

        time.sleep(0.1)
        #if idx > 10:
        #    break
