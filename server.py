import asyncio
import logging
import multiprocessing
from asyncio import Queue

import websockets

logging.basicConfig(level=logging.INFO)


class WebSocketServer(multiprocessing.Process):

    clients = set()

    def __init__(self, address, port):
        multiprocessing.Process.__init__(self)
        self.address = address
        self.port = port
        self.msg_queue = Queue()
        self.dash_client_path = "/dash_client"

    def run(self):
        asyncio.run(self.main())

    async def main(self):
        async with websockets.serve(self.ws_handler, self.address, self.port) as self.websocket:
            await asyncio.Future()

    async def unregister(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.remove(ws)
        logging.info(f'{ws.remote_address[0]}:{ws.remote_address[1]}{ws.path} disconnects')

    async def register(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        logging.info(f'{ws.remote_address[0]}:{ws.remote_address[1]}{ws.path} connects')
        if ws.path == self.dash_client_path:
            await self.send_from_queue()
            logging.info(f'all messages was sent to {ws.remote_address[0]}:{ws.remote_address[1]}{ws.path}')

    async def send_from_queue(self) -> None:
        while self.msg_queue.qsize() > 0:
            dash_client_paths = [client.path for client in self.clients if client.path == self.dash_client_path]
            if len(dash_client_paths) > 0:
                msg = self.msg_queue.get_nowait()
                await asyncio.wait([asyncio.create_task(client.send(msg))
                                    for client in self.clients
                                    if client.path == self.dash_client_path])
            else:
                break

    async def send_to_clients(self, message: str) -> None:
        await self.msg_queue.put(message)
        await self.send_from_queue()

    async def ws_handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        await self.register(ws)
        try:
            await self.distribute(ws)
        except websockets.ConnectionClosedError:
            logging.info(f'{ws.remote_address[0]}:{ws.remote_address[1]}{ws.path} closed by interrupt')
        finally:
            await self.unregister(ws)

    async def distribute(self, ws: websockets.WebSocketServerProtocol) -> None:
        async for message in ws:
            await self.send_to_clients(message)

    def stop(self):
        self.websocket.ws_server.close()
