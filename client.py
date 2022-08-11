import asyncio
import logging

import websockets

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
        return self.loop.run_until_complete(self.__async__command(message))

    async def __async__command(self, message):
        await self.ws.send(message)
        # return await self.ws.recv()
