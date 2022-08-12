from .dash_viz.app import DashApp
from .websocket.server import WebSocketServer


class DashVisualizer:
    def __init__(self, name, address, ws_port, dash_port, ws_url):
        self.ws_thread = WebSocketServer(address, ws_port)
        self.dash_thread = DashApp(title=name,
                                   host=address,
                                   port=dash_port,
                                   websocket_url=ws_url)
        self.ws_thread.start()
        self.dash_thread.start()
