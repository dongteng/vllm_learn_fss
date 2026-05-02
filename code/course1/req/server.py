# server.py - ZeroMQ Echo 服务端 (REP)
import zmq # 导入 ZeroMQ 库 


def echo_server():
    context = zmq.Context() #Context 是 ZMQ 的运行环境 + 资源管理器,它管理IO线程、socket的底层资源
    socket = context.socket(zmq.REP)  # REP socket  #设置通信语义是REP
    socket.bind("tcp://*:5555")  #使用TCP协议  绑定到本机所有IP的5555端口

    print("Echo 服务端启动，等待客户端连接...")

    while True:
        # 等待接收消息
        message = socket.recv()  # 阻塞等待
        print(f"收到消息: {message.decode()}") #message是字符串 需要解码成字符串

        message= message+' ,这里是服务端'.encode("utf-8") #将字符串编码成字节串
        # 回显相同的消息
        socket.send(message)


if __name__ == "__main__":
    try:
        echo_server()
    except KeyboardInterrupt:
        print("\n服务端关闭。")
