from socket import socket
from typing import List, Optional, Callable, Tuple


def eprint(*args, **kwargs):
    from sys import stderr
    print(*args, file=stderr, **kwargs)  # TODO Is there some warn(...) equivalent function?


def static_vars(**kwargs):
    """
    Decorator hack for introducing local static variables.
    :param kwargs: The declarations of the static variables like "foo=42".
    :return: The decorated function.
    """

    def decorate(func):
        """
        Decorates the given function with local static variables based on kwargs.
        :param func: The function to decorate.
        :return: The decorated function.
        """
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def create_server(port: int) -> socket:
    from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, SO_REUSEADDR
    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(1)  # FIXME Determine appropriate value
    return server_socket


def accept_at_server(server_socket: socket, on_accept: Callable[[socket, Tuple[str, int]], None]) -> None:
    while True:
        conn, addr = server_socket.accept()
        print(str(server_socket.getsockname()) + " accepted " + str(addr))
        on_accept(conn, addr)


def create_client(server_host: str, server_port: int) -> socket:
    from socket import AF_INET, SOCK_STREAM
    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect((server_host, server_port))
    return client_socket


@static_vars(block_size=1024)  # FIXME Determine optimal buffer size
def _recv_all(sock: socket, timeout: float) -> bytes:
    from select import select
    response = b""
    while True:  # Pseudo "do_while"-loop
        sread, _, _ = select([sock], [], [], timeout)
        last_chunk = sread[0].recv(_recv_all.block_size) if sread else b""
        response = response + last_chunk
        if len(last_chunk) < _recv_all.block_size:
            break
    return response


def process_message(waiting_socket: socket, handle_message: Callable[[bytes, List[bytes]], bytes],
                    timeout: float = 0.2) -> None:
    from client.aiExchangeMessages_pb2 import Num
    from select import select
    # FIXME How to recover failures?
    addr = str(waiting_socket.getsockname())
    import threading
    # print(str(threading.current_thread().ident) + " " + addr + " begin while")
    waiting_socket.send(b"ready for action")
    action = waiting_socket.recv(1024)  # FIXME Determine optimal buffer size
    waiting_socket.send(b"ready for num data")
    num_data = Num()
    # FIXME Transmissions of 0 or Void which is encoded an empty string slows down
    sread, _, _ = select([waiting_socket], [], [], timeout)
    got = sread[0].recv(1024) if sread else b""
    num_data.ParseFromString(got)  # FIXME Determine optimal buffer size
    data = []
    for _ in range(0, num_data.num):
        waiting_socket.send(b"ready for data")
        data.append(_recv_all(waiting_socket, timeout))

    result = handle_message(action, data)

    waiting_socket.send(result)
    result_received = waiting_socket.recv(15)
    if not result_received == b"result received":
        eprint("Expected \"result received\". Got \"" + result_received.decode() + "\".")
    # print(str(threading.current_thread().ident) + " " + addr + " end while")


def process_messages(waiting_socket: socket, handle_message: Callable[[bytes, List[bytes]], bytes],
                     timeout: float = 0.2) -> None:
    # FIXME How to recover failures?
    try:
        while True:
            process_message(waiting_socket, handle_message, timeout)
    except (ConnectionAbortedError, ConnectionResetError):
        eprint("The socket " + str(waiting_socket.getsockname()) + " was closed.")


def send_message(sending_socket: socket, action: bytes, data: List[bytes],
                 timeout: Optional[float] = 0.5) -> Optional[bytes]:
    from client.aiExchangeMessages_pb2 import Num
    try:
        action_ready = sending_socket.recv(16)
        if action_ready == b"ready for action":
            sending_socket.send(action)
            num_data_ready = sending_socket.recv(18)
            if num_data_ready == b"ready for num data":
                num_data = Num()
                num = len(data)
                num_data.num = -1 if num == 0 else num  # NOTE Send -1 instead of 0 because of performance reasons
                sending_socket.send(num_data.SerializeToString())
                for datum in data:
                    data_ready = sending_socket.recv(14)
                    if data_ready == b"ready for data":
                        sending_socket.send(datum)
                    else:
                        eprint("Expected \"ready for data\". Got \"" + data_ready.decode() + "\".")
                        result = None
                        break
                else:
                    # FIXME Determine appropriate timeout for result
                    result = _recv_all(sending_socket, timeout)
                    sending_socket.send(b"result received")
            else:
                eprint("Expected \"ready for num data\". Got \""
                       + num_data_ready.decode() + "\".")
                result = None
        else:
            eprint(str(
                sending_socket.getsockname()) + " expected \"ready for action\". Got \"" + action_ready.decode() + "\".")
            result = None
    except ConnectionResetError as ex:
        eprint("Sending message failed due to a ConnectionResetError")
        eprint(str(ex))
        result = None
    return result
