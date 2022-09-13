import asyncio
import json
import logging
import multiprocessing

from asyncio import Queue
from typing import Union

import websockets

logging.basicConfig(level=logging.INFO)


class WebSocketServer(multiprocessing.Process):
    clients = set()

    def __init__(self, address: str, port: Union[int, str]):
        multiprocessing.Process.__init__(self)
        self.address = address
        self.port = port
        self.msg_queue = Queue()
        self.dash_client_path = "/dash_client"

    async def send_from_queue(self):
        while not self.msg_queue.empty():
            dash_clients = [client for client in self.clients if client.path == self.dash_client_path]
            if len(dash_clients) > 0:
                message = self.msg_queue.get_nowait()
                for dash_client in dash_clients:
                    await dash_client.msg_queue.put(message)
            else:
                break

    def run(self):
        try:
            asyncio.run(self.main())
        except KeyboardInterrupt:
            try:
                self.stop()
            finally:
                logging.info(f'server: closed by keyboard interrupt')

    async def main(self):
        async with websockets.serve(self.ws_handler, self.address, self.port,
                                    max_size=None, max_queue=None) as self.websocket:
            await asyncio.Future()

    async def unregister(self, ws: websockets.WebSocketServerProtocol):
        self.clients.remove(ws)
        logging.info(f'server: {ws.remote_address[0]}:{ws.remote_address[1]}{ws.path} disconnects')

    async def register(self, ws: websockets.WebSocketServerProtocol):
        if ws.path == self.dash_client_path and not hasattr(ws, 'msg_queue'):
            ws.msg_queue = Queue()
        self.clients.add(ws)
        await self.send_from_queue()
        logging.info(f'server: {ws.remote_address[0]}:{ws.remote_address[1]}{ws.path} connects')

    async def ws_handler(self, ws: websockets.WebSocketServerProtocol):
        await self.register(ws)
        try:
            if ws.path == self.dash_client_path:
                await self.dash_distribute(ws)
            else:
                await self.distribute(ws)
        except websockets.ConnectionClosedError:
            logging.info(f'server: {ws.remote_address[0]}:{ws.remote_address[1]}{ws.path} closed as it was lost')
        finally:
            await self.unregister(ws)

    async def dash_distribute(self, ws: websockets.WebSocketServerProtocol):
        async for _ in ws:
            message = await ws.msg_queue.get()
            await ws.send(message)

    async def distribute(self, ws: websockets.WebSocketServerProtocol):
        async for message in ws:
            dash_client_exist = False
            for client in self.clients:
                if client.path == self.dash_client_path:
                    dash_client_exist = True
                    await client.msg_queue.put(message)

            if not dash_client_exist:
                await self.msg_queue.put(message)

    def stop(self):
        self.websocket.ws_server.close()
