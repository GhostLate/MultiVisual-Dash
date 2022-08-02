import asyncio
import logging
import multiprocessing

import websockets

logging.basicConfig(level=logging.INFO)


class WebSocketServer(multiprocessing.Process):
    clients = set()

    def __init__(self, address, port):
        multiprocessing.Process.__init__(self)
        self.address = address
        self.port = port

    def run(self):
        asyncio.run(self.main())

    async def main(self):
        async with websockets.serve(self.ws_handler, self.address, self.port) as self.websocket:
            await asyncio.Future()

    async def register(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        logging.info(f'{ws.remote_address} connects')

    async def unregister(self, ws: websockets.WebSocketServerProtocol) -> None:
        self.clients.remove(ws)
        logging.info(f'{ws.remote_address} disconnects')

    async def send_to_clients(self, message: str) -> None:
        if self.clients:
            await asyncio.wait([asyncio.create_task(client.send(message)) for client in self.clients])

    async def ws_handler(self, ws: websockets.WebSocketServerProtocol) -> None:
        await self.register(ws)
        try:
            await self.distribute(ws)
        except websockets.ConnectionClosedError:
            logging.info(f'{ws.remote_address} closed by interrupt')
        finally:
            await self.unregister(ws)

    async def distribute(self, ws: websockets.WebSocketServerProtocol) -> None:
        async for message in ws:
            await self.send_to_clients(message)

    def stop(self):
        self.websocket.ws_server.close()
