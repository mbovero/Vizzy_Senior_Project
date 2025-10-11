#!/usr/bin/env python3
import sys
import json
import socket

HOST = "raspberrypi.local"  # or the Pi's IP, e.g., "192.168.1.42"
PORT = 5005
AUTH_TOKEN = None  # e.g., "secret123" if enabled on server

def send(cmd: dict):
    if AUTH_TOKEN is not None:
        cmd["token"] = AUTH_TOKEN
    line = (json.dumps(cmd) + "\n").encode("utf-8")
    with socket.create_connection((HOST, PORT), timeout=5) as s:
        s.sendall(line)
        resp = s.recv(4096).decode("utf-8").strip()
        print(resp)

def usage():
    print("Usage:")
    print("  Center all:        pc_servo_client.py center")
    print("  Stop all:          pc_servo_client.py stop")
    print("  Set one:           pc_servo_client.py set <btm|mid|top> <us>")
    print("  Get state:         pc_servo_client.py get")
    print("Example:")
    print("  pc_servo_client.py set mid 1500")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    op = sys.argv[1]
    if op == "center":
        send({"cmd": "center"})
    elif op == "stop":
        send({"cmd": "stop"})
    elif op == "get":
        send({"cmd": "get"})
    elif op == "set" and len(sys.argv) == 4:
        target = sys.argv[2]
        us = int(sys.argv[3])
        send({"cmd": "set", "target": target, "us": us})
    else:
        usage()
        sys.exit(1)
