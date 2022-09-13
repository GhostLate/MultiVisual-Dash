from multi_visual_dash.dash_visualizer import DashVisualizer
from multi_visual_dash.dataloaders.dynamic_objects_dataset.visualizer import DynamicObjectsVisualizer
from multi_visual_dash.websocket.client import WebSocketClient

if __name__ == "__main__":
    address = "localhost"
    ws_port = 4000
    websocket_url = f"ws://{address}:{ws_port}"
    visualizer_server = DashVisualizer('Dynamic Objects PC', address, ws_port, 8000, f"{websocket_url}/dash_client")
    visualizer_client = WebSocketClient(websocket_url)

    waymo_data_loader = DynamicObjectsVisualizer('./data/dynamic_objects_dataset/waymo/', 200, 10000, True)

    for idx, plot_data in enumerate(waymo_data_loader()):
        visualizer_client.send(dict(plot_data))
