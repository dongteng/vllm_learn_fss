# pull.py - PULL 端：接收并处理流式 token
import zmq


def main():
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:7777")  # 绑定到端口 7777

    print("PULL: Waiting for tokens...")

    while True:
        message = receiver.recv_string()
        if message == "END":
            print("PULL: Received end of stream.")
            break
        print(f"PULL: Received token '{message}'")

    receiver.close()
    context.term()


if __name__ == "__main__":
    main()
