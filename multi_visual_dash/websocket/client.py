import asyncio
import json
import logging
import multiprocessing
from multiprocessing import Queue as mQueue

import websockets

from multi_visual_dash.websocket.utils import NumpyEncoder

logging.basicConfig(level=logging.INFO)


class WebSocketClient(multiprocessing.Process):
    ws: websockets.WebSocketClientProtocol

    def __init__(self, websocket_url: str, sleep_time: float = 1):
        multiprocessing.Process.__init__(self)
        self.websocket_url = websocket_url
        self.msg_queue = mQueue()
        self.last_message = None
        self.sleep_time = sleep_time
        self.start()

    def run(self):
        try:
            asyncio.run(self.main())
        except KeyboardInterrupt:
            logging.info(f'client: closed by keyboard interrupt')

    async def main(self):
        logging.info(f"client: Attempting connection to: {self.websocket_url}")
        while True:
            try:
                async with websockets.connect(self.websocket_url, max_size=None) as self.ws:
                    logging.info("client: Connected")
                    try:
                        await self.send_handler()
                    except websockets.ConnectionClosed:
                        logging.info(f'client: {self.websocket_url} closed')
            except Exception as ex:
                logging.debug(f'client: trying to connect to server')
                logging.debug(f'client : error: {ex}')
            finally:
                await asyncio.sleep(self.sleep_time)

    async def send_handler(self):
        while True:
            if self.last_message is None:
                self.last_message = self.msg_queue.get()
            await self.ws.send(self.last_message)
            self.last_message = None

    def send(self, message):
        if not isinstance(message, str):
            message = json.dumps(message, cls=NumpyEncoder)
        self.msg_queue.put(message)
