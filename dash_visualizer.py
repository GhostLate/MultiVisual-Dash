from dash_app import DashApp
from server import WebSocketServer


class DashVisualizer:
    def __init__(self, graph_name, address, ws_port, dash_port, ws_url):
        self.ws_thread = WebSocketServer(address, ws_port)
        self.dash_thread = DashApp(host=address,
                                   port=dash_port,
                                   plot_name=graph_name,
                                   websocket_url=ws_url)
        self.ws_thread.start()
        self.dash_thread.start()
