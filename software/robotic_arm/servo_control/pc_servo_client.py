#!/usr/bin/env python3
"""
PC Client for the Pi Servo Server
Usage examples:
  python pc_servo_client.py --host 192.168.1.42 get
  python pc_servo_client.py --host 192.168.1.42 set mid 1700
  python pc_servo_client.py --host 192.168.1.42 center
  python pc_servo_client.py --host 192.168.1.42 stop

You can also set env vars:
  SERVO_HOST=192.168.1.42 SERVO_PORT=5005 SERVO_TOKEN=secret123
"""

import sys
import json
import socket
import argparse
import os

DEFAULT_HOST = os.environ.get("SERVO_HOST", "raspberrypi.local")
DEFAULT_PORT = int(os.environ.get("SERVO_PORT", "5005"))
AUTH_TOKEN   = os.environ.get("SERVO_TOKEN")  # optional

def send(cmd: dict, host: str, port: int):
    if AUTH_TOKEN:
        cmd["token"] = AUTH_TOKEN
    line = (json.dumps(cmd) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(line)
        resp = s.recv(4096).decode("utf-8").strip()
        print(resp)

def main():
    p = argparse.ArgumentParser(description="PC → Pi servo client")
    p.add_argument("op", choices=["center","stop","get","set"], help="operation")
    p.add_argument("args", nargs="*", help="extra args for 'set'")
    p.add_argument("--host", default=DEFAULT_HOST, help="Pi host/IP")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    a = p.parse_args()

    if a.op == "center":
        send({"cmd": "center"}, a.host, a.port)
    elif a.op == "stop":
        send({"cmd": "stop"}, a.host, a.port)
    elif a.op == "get":
        send({"cmd": "get"}, a.host, a.port)
    elif a.op == "set":
        if len(a.args) != 2:
            print("Usage: set <btm|mid|top> <us>")
            sys.exit(1)
        target, us = a.args[0], int(a.args[1])
        send({"cmd": "set", "target": target, "us": us}, a.host, a.port)

if __name__ == "__main__":
    main()
