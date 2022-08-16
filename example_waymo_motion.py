import time

from dash_visualizer import DashVisualizer
from dataloaders.waymo.motion_dataloader import WaymoMotionDataLoader
from websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4002
    websocket_url = f"ws://{address}:{ws_port}"

    waymo_data_loader = WaymoMotionDataLoader("./data/validation_tfexample.tfrecord-00000-of-00150")
    visualizer_server = DashVisualizer('Waymo Motion', address, ws_port, 8003, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)
    i = 0

    for plot_data in waymo_data_loader():
        visualizer_client.send(plot_data)

        time.sleep(1)
        i += 1
        if i > 1:
            break
