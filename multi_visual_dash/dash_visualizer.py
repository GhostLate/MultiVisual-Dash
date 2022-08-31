from typing import Union

from multi_visual_dash.dash_viz.app import DashApp
from multi_visual_dash.websocket.server import WebSocketServer


class DashVisualizer:
    def __init__(self, name: str, address: str, ws_port: Union[int, str], dash_port: Union[int, str], ws_url: str):
        self.ws_thread = WebSocketServer(address, ws_port)
        self.dash_thread = DashApp(title=name,
                                   host=address,
                                   port=dash_port,
                                   websocket_url=ws_url)
        self.ws_thread.start()
        self.dash_thread.start()
