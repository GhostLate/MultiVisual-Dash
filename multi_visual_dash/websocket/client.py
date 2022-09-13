import asyncio
import json
import logging
from queue import Queue

import websockets
from multi_visual_dash.websocket.utils import NumpyEncoder

logging.basicConfig(level=logging.INFO)


class WebSocketClient:
    def __init__(self, websocket_url: str):
        self.ws = None
        self.websocket_url = websocket_url
        self.loop = asyncio.get_event_loop()
        self.msg_queue = Queue()
        self.loop.run_until_complete(self.__async__connect())

    async def __async__connect(self):
        logging.info(f"client: Attempting connection to: {self.websocket_url}")
        self.ws = await websockets.connect(self.websocket_url, max_size=None)
        logging.info("client: Connected")

    def send(self, message):
        if not isinstance(message, str):
            message = json.dumps(message, cls=NumpyEncoder)
        self.msg_queue.put(message)
        return self.loop.run_until_complete(self.send_from_queue())

    async def send_from_queue(self):
        while self.msg_queue.qsize() > 0:
            message = self.msg_queue.get_nowait()
            try:
                await self.ws.send(message)
            except websockets.ConnectionClosedError:
                logging.info(f'client: {self.ws.remote_address[0]}:{self.ws.remote_address[1]}{self.ws.path} closed by interrupt')



