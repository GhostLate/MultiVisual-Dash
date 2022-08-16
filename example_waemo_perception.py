import time

from dataloaders.waymo.perception_dataloader import WaymoPerceptionDataLoader
from dash_visualizer import DashVisualizer
from websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4002
    websocket_url = f"ws://{address}:{ws_port}"

    waymo_data_loader = WaymoPerceptionDataLoader('./data/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord')
    visualizer_server = DashVisualizer('Waymo Point Cloud', address, ws_port, 8003, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    i = 0
    for plot_data in waymo_data_loader():
        visualizer_client.send(plot_data)

        time.sleep(2)
        i += 1
        if i > 1:
            break

