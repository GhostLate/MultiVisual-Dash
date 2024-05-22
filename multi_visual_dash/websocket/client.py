import asyncio
import logging
import multiprocessing

import websockets

from multi_visual_dash.dash_viz.data import DashMessage
from multi_visual_dash.websocket.utils import compress_message

logging.basicConfig(level=logging.INFO)


class WebSocketClient(multiprocessing.Process):
    ws: websockets.WebSocketClientProtocol

    def __init__(self, websocket_url: str, sleep_time: float = 1):
        multiprocessing.Process.__init__(self)
        self.websocket_url = websocket_url
        self.msg_queue = multiprocessing.Queue()
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
        last_message = None
        while True:
            if last_message is None:
                last_message = self.msg_queue.get()
            await self.ws.send(last_message)
            last_message = None
            await asyncio.sleep(0)

    def send(self, message):
        if isinstance(message, DashMessage):
            message = dict(message)
        if not isinstance(message, dict):
            logging.error("client: the message doesn't have a 'dict' instance")
        compressed_message = compress_message(message)
        self.msg_queue.put(compressed_message)
