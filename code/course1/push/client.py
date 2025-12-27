# push.py - PUSH 端：模拟推理过程，逐个 token 推送
import zmq
import time


def generate_tokens():
    # 模拟逐个生成 token
    for token in ["Hello", " world", ",", " how", " are", " you", "?"]:
        yield token
        time.sleep(0.1)  # 模拟生成延迟


def main():
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:7777")

    print("PUSH: Sending tokens...")

    for token in generate_tokens():
        sender.send_string(token)

    # 发送结束标记
    sender.send_string("END")
    print("PUSH: All tokens sent.")

    sender.close()
    context.term()


if __name__ == "__main__":
    main()
