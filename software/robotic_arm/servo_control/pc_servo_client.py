#!/usr/bin/env python3
"""
PC Client for the Pi Servo Server (port 65432)
Usage:
  python pc_servo_client.py --host 10.120.39.241 get
  python pc_servo_client.py --host 10.120.39.241 set mid 1700
  python pc_servo_client.py --host 10.120.39.241 center
  python pc_servo_client.py --host 10.120.39.241 stop
"""

import sys, json, socket, argparse, os

DEFAULT_HOST = os.environ.get("SERVO_HOST", "10.120.39.241")
DEFAULT_PORT = int(os.environ.get("SERVO_PORT", "65432"))
AUTH_TOKEN   = os.environ.get("SERVO_TOKEN")

def send(cmd, host, port):
    if AUTH_TOKEN:
        cmd["token"] = AUTH_TOKEN
    line = (json.dumps(cmd) + "\n").encode()
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(line)
        print(s.recv(4096).decode().strip())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("op", choices=["center","stop","get","set"])
    p.add_argument("args", nargs="*")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = p.parse_args()

    if a.op == "center": send({"cmd": "center"}, a.host, a.port)
    elif a.op == "stop": send({"cmd": "stop"}, a.host, a.port)
    elif a.op == "get":  send({"cmd": "get"}, a.host, a.port)
    elif a.op == "set":
        if len(a.args) != 2:
            print("Usage: set <btm|mid|top> <us>"); sys.exit(1)
        send({"cmd":"set","target":a.args[0],"us":int(a.args[1])}, a.host, a.port)

if __name__ == "__main__":
    main()
