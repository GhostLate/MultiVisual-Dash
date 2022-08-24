import asyncio
import json
import logging

import websockets

from multi_visual_dash.websocket.utils import NumpyEncoder

logging.basicConfig(level=logging.INFO)


class WebSocketClient:
    def __init__(self, websocket_url):
        self.ws = None
        self.websocket_url = websocket_url
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.__async__connect())

    async def __async__connect(self):
        logging.info(f"Attempting connection to: {self.websocket_url}")
        self.ws = await websockets.connect(self.websocket_url, max_size=None)
        logging.info("Connected")

    def send(self, message):
        if not isinstance(message, str):
            message = json.dumps(message, cls=NumpyEncoder)
        return self.loop.run_until_complete(self.__async__command(message))

    async def __async__command(self, message):
        await self.ws.send(message)
