# server.py - ZeroMQ Echo 服务端 (REP)
import zmq


def echo_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP socket
    socket.bind("tcp://*:5555")  # 绑定到所有IP的5555端口

    print("Echo 服务端启动，等待客户端连接...")

    while True:
        # 等待接收消息
        message = socket.recv()  # 阻塞等待
        print(f"收到消息: {message.decode()}")

        # 回显相同的消息
        socket.send(message)


if __name__ == "__main__":
    try:
        echo_server()
    except KeyboardInterrupt:
        print("\n服务端关闭。")
