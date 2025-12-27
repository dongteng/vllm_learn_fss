# dealer_router_example.py
import time
import zmq
import threading
import uuid


# -----------------------------
# 客户端（Client） - 使用 DEALER
# -----------------------------
def client(client_id):
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)

    # 为每个客户端设置唯一身份标识（ZeroMQ 会自动使用此身份进行路由）
    identity = f"Client-{client_id}".encode("utf-8")
    socket.setsockopt(zmq.IDENTITY, identity)

    socket.connect("tcp://localhost:6666")
    print(f"[Client {client_id}] 启动，连接到引擎...")

    # 发送多个异步请求
    for i in range(5):
        request = {
            "msg_id": str(uuid.uuid4()),
            "query": f"请求 {i} 来自 {client_id}",
            "timestamp": time.time(),
        }
        socket.send_json(request)
        print(f"[Client {client_id}] 发送请求: {request['msg_id']}")

        # 不等待响应，继续发送下一个请求（异步）
        time.sleep(0.1)

    # 接收响应（异步接收，顺序不一定与发送一致）
    for _ in range(5):
        try:
            response = socket.recv_json()  # 非阻塞接收
            print(f"[Client {client_id}] 收到响应: {response}")
        except zmq.Again:
            time.sleep(2)
            continue

    socket.close()


# -----------------------------
# 推理引擎（Engine） - 使用 ROUTER + Poller
# -----------------------------
def engine():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)  # 接收客户端请求
    frontend.bind("tcp://*:6666")

    print("推理引擎启动，监听端口 6666...")

    def process_inference(data):
        """模拟异步推理任务"""
        time.sleep(0.5)  # 模拟计算延迟
        return {
            "status": "success",
            "result": f"推理完成: {data['query']}",
            "processed_at": time.time(),
        }

    # 创建 Poller 并注册 frontend 套接字
    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)  # 监听是否有消息可读

    try:
        while True:
            # 阻塞最多 1 秒，检查是否有事件就绪
            # 返回格式: {socket: event_mask}
            socks = dict(poller.poll(timeout=1000))  # 超时单位：毫秒

            # 检查 frontend 是否有来自客户端的消息
            if frontend in socks and socks[frontend] == zmq.POLLIN:
                multipart = frontend.recv_multipart()
                identity = multipart[0]
                message = multipart[-1]
                request = zmq.utils.jsonapi.loads(message)

                print(f"引擎收到来自 {identity.decode()} 的请求: {request['msg_id']}")

                # 模拟处理（实际中可转发给 backend 或线程池）
                result = process_inference(request)

                # 构造响应并返回
                response = {
                    "msg_id": request["msg_id"],
                    "reply": result,
                    "from_engine": "Engine-0",
                }
                frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])

            # === 后续可在此添加其他事件处理逻辑 ===
            # 例如：
            # - 监听 backend 返回结果
            # - 处理心跳或控制命令
            # - 清理超时请求等

    except KeyboardInterrupt:
        print("\n引擎关闭.")
    finally:
        frontend.close()


# -----------------------------
# 主程序：启动服务端和多个客户端
# -----------------------------
if __name__ == "__main__":
    # 启动引擎线程
    engine_thread = threading.Thread(target=engine, daemon=True)
    engine_thread.start()

    time.sleep(1)  # 等待引擎启动

    # 启动多个客户端
    client_threads = []
    for i in range(3):
        t = threading.Thread(target=client, args=(i,), daemon=True)
        t.start()
        client_threads.append(t)

    try:
        # 等待所有客户端完成
        for t in client_threads:
            t.join()
        time.sleep(10)
    except KeyboardInterrupt:
        print("\n主程序退出.")
