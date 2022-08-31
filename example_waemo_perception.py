import time

from multi_visual_dash.dash_visualizer import DashVisualizer
from multi_visual_dash.dash_viz.data import SceneCamera
from multi_visual_dash.dataloaders.waymo.perception.visualizer import WaymoPerceptionVisualizer
from multi_visual_dash.websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4000
    websocket_url = f"ws://{address}:{ws_port}"

    waymo_data_visualizer = WaymoPerceptionVisualizer('./data/perception/', center_data=True)
    visualizer_server = DashVisualizer('Waymo Point Cloud', address, ws_port, 8000, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    scene_camera = SceneCamera()
    scene_camera.up.z = 1
    scene_camera.eye.x = -0.5
    scene_camera.eye.z = 0.1
    scene_camera.center.z = 0.05

    time.sleep(1)
    for idx, plot_data in enumerate(waymo_data_visualizer()):
        if idx == 0 and waymo_data_visualizer.center_data:
            plot_data.scene_camera = scene_camera

        visualizer_client.send(dict(plot_data))
        time.sleep(0.1)
