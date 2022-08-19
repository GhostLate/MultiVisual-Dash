import time

from dash_visualizer import DashVisualizer
from dash_viz.data import SceneCamera
from dataloaders.waymo.perception_dataloader import WaymoPerceptionDataLoader
from websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4002
    websocket_url = f"ws://{address}:{ws_port}"

    waymo_data_loader = WaymoPerceptionDataLoader('./data/individual_files_validation_segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord')
    visualizer_server = DashVisualizer('Waymo Point Cloud', address, ws_port, 8002, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    scene_camera = SceneCamera()
    scene_camera.up.z = 1
    scene_camera.eye.x = -0.5
    scene_camera.eye.z = 0.1
    scene_camera.center.z = 0.05

    time.sleep(1)
    for idx, plot_data in enumerate(waymo_data_loader()):
        if idx == 0:
            plot_data.scene_camera = scene_camera

        visualizer_client.send(dict(plot_data))
        time.sleep(.1)
