# laptop/dispatch.py
"""
Command dispatcher for sending primitive commands to RPi one at a time.
Waits for confirmation before sending the next command.
"""

import time
import queue
import socket
from typing import List, Dict, Optional

from ..shared import protocol as P
from ..shared import config as C
from ..shared.jsonl import send_json


class CommandDispatcher:
    """
    Sends primitive commands to RPi one at a time.
    Waits for confirmation before sending next command.
    """
    
    def __init__(self, sock, cmd_complete_queue: queue.Queue):
        """
        Parameters
        ----------
        sock : socket.socket
            TCP connection to RPi
        cmd_complete_queue : queue.Queue
            Queue that receives CMD_COMPLETE messages from receiver thread
        """
        self.sock = sock
        self._cmd_complete_q = cmd_complete_queue
        self.timeout = C.PRIMITIVE_CMD_TIMEOUT
    
    def send_command(self, cmd_dict: Dict) -> bool:
        """
        Send a single primitive command to RPi and wait for confirmation.
        
        Parameters
        ----------
        cmd_dict : Dict
            Command with protocol constants, e.g.:
            {"cmd": P.CMD_MOVE_TO, "x": 1200, "y": 800, "z": 500}
            {"cmd": P.CMD_GRAB}
            {"cmd": P.CMD_ROT_YAW, "angle": 45.0}
        
        Returns
        -------
        bool
            True if command succeeded, False otherwise
        """
        try:
            # Send to RPi
            send_json(self.sock, cmd_dict)
            cmd_name = cmd_dict.get("cmd", "UNKNOWN")
            print(f"[Dispatcher] Sent: {cmd_name}")
            
            # Wait for confirmation
            status = self._wait_for_confirmation(cmd_name)
            
            if status == "success":
                print(f"[Dispatcher] Confirmed: {cmd_name}")
                return True
            elif status == "error":
                print(f"[Dispatcher] Error: {cmd_name}")
                return False
            else:
                print(f"[Dispatcher] Timeout: {cmd_name}")
                return False
                
        except Exception as e:
            print(f"[Dispatcher] Exception sending command: {e}")
            return False
    
    def execute_primitives(self, primitives: List[Dict]) -> bool:
        """
        Execute a list of primitive commands one at a time.
        
        Parameters
        ----------
        primitives : List[Dict]
            List of command dicts with protocol format, e.g.:
            [
                {"cmd": P.CMD_RELEASE},
                {"cmd": P.CMD_MOVE_TO, "x": 1200, "y": 800, "z": 500},
                {"cmd": P.CMD_GRAB}
            ]
        
        Returns
        -------
        bool
            True if all commands succeeded, False if any failed
        """
        print(f"[Dispatcher] Executing {len(primitives)} command(s)")
        
        for i, cmd in enumerate(primitives, 1):
            cmd_name = cmd.get("cmd", "UNKNOWN")
            print(f"[Dispatcher] [{i}/{len(primitives)}] {cmd_name}")
            
            if not self.send_command(cmd):
                print(f"[Dispatcher] FAILED at command {i}")
                return False
        
        print(f"[Dispatcher] All {len(primitives)} commands completed successfully")
        return True
    
    def _wait_for_confirmation(self, expected_cmd: str) -> str:
        """
        Wait for CMD_COMPLETE message from RPi.
        
        Parameters
        ----------
        expected_cmd : str
            The command we're waiting for (e.g., "MOVE_TO", "GRAB")
        
        Returns
        -------
        str
            "success", "error", or "timeout"
        """
        deadline = time.time() + self.timeout
        
        while time.time() < deadline:
            timeout_remaining = deadline - time.time()
            if timeout_remaining <= 0:
                break
            
            try:
                # Wait for message with timeout
                msg = self._cmd_complete_q.get(timeout=min(timeout_remaining, 0.1))
                
                # Check if this is the confirmation we're waiting for
                if msg.get("type") == P.TYPE_CMD_COMPLETE:
                    if msg.get("cmd") == expected_cmd:
                        status = msg.get("status", "unknown")
                        if status == "success":
                            return "success"
                        else:
                            # Log error message if provided
                            error_msg = msg.get("message", "Unknown error")
                            print(f"[Dispatcher] RPi reported error: {error_msg}")
                            return "error"
                    else:
                        # Wrong command confirmation, put it back and continue
                        print(f"[Dispatcher] Warning: Received confirmation for {msg.get('cmd')} "
                              f"but waiting for {expected_cmd}")
                        # Note: We don't put it back as it's likely a stale message
                        continue
                
            except queue.Empty:
                # No message yet, continue waiting
                continue
        
        # Timeout reached
        print(f"[Dispatcher] Timeout waiting for {expected_cmd} confirmation ({self.timeout}s)")
        return "timeout"
    
    def execute_ik_commands(self, ik_commands: List[Dict]) -> bool:
        """
        Execute a list of IK commands by sending them as text to the arm server.
        Includes a 2-second delay between commands.
        
        Parameters
        ----------
        ik_commands : List[Dict]
            List of IK command dicts with format:
            {"type": "ik", "x": 0.3, "y": 0.2, "z": 0.125, "pitch_rad": -1.571, "yaw_rad": 0.0, "claw": "O"}
        
        Returns
        -------
        bool
            True if all commands succeeded, False if any failed
        """
        print(f"[Dispatcher] Executing {len(ik_commands)} IK command(s)")
        
        for i, cmd in enumerate(ik_commands, 1):
            if cmd.get("type") != "ik":
                print(f"[Dispatcher] Warning: Skipping non-IK command: {cmd}")
                continue
            
            # Format: "ik x y z pitch_rad yaw_rad O|C\n"
            x = cmd.get("x", 0.0)
            y = cmd.get("y", 0.0)
            z = cmd.get("z", 0.0)
            pitch_rad = cmd.get("pitch_rad", 0.0)
            yaw_rad = cmd.get("yaw_rad", 0.0)
            claw = cmd.get("claw", "O")
            
            cmd_text = f"ik {x:.6f} {y:.6f} {z:.6f} {pitch_rad:.6f} {yaw_rad:.6f} {claw}\n"
            print(f"[Dispatcher] [{i}/{len(ik_commands)}] Sending IK command: {cmd_text.strip()}")
            
            try:
                # Send text command directly to socket
                self.sock.sendall(cmd_text.encode("utf-8"))
                
                # Wait for ACK (arm server sends "ACK ik\n" or "ERR ...\n")
                ack = self._wait_for_ik_ack()
                
                if ack == "success":
                    print(f"[Dispatcher] IK command {i} confirmed")
                elif ack == "error":
                    print(f"[Dispatcher] IK command {i} FAILED")
                    return False
                else:
                    print(f"[Dispatcher] IK command {i} TIMEOUT")
                    return False
                
                # Add 2-second delay between commands (except after the last one)
                if i < len(ik_commands):
                    print(f"[Dispatcher] Waiting 2 seconds before next command...")
                    time.sleep(2.0)
                    
            except Exception as e:
                print(f"[Dispatcher] Exception sending IK command: {e}")
                return False
        
        print(f"[Dispatcher] All {len(ik_commands)} IK commands completed successfully")
        return True
    
    def _wait_for_ik_ack(self) -> str:
        """
        Wait for ACK/ERR response from arm server for IK command.
        Responses are received by the receiver thread and put in the queue as strings.
        
        Returns
        -------
        str
            "success", "error", or "timeout"
        """
        deadline = time.time() + self.timeout
        
        while time.time() < deadline:
            timeout_remaining = deadline - time.time()
            if timeout_remaining <= 0:
                break
            
            try:
                # Wait for message from receiver thread (queue contains strings for text protocol)
                msg = self._cmd_complete_q.get(timeout=min(timeout_remaining, 0.1))
                
                # Handle text protocol responses (strings)
                if isinstance(msg, str):
                    if msg.startswith("ACK"):
                        return "success"
                    elif msg.startswith("ERR"):
                        print(f"[Dispatcher] Server error: {msg}")
                        return "error"
                    # Otherwise continue waiting
                
                # Handle JSON protocol responses (dicts) - backward compatibility
                elif isinstance(msg, dict):
                    if msg.get("type") == P.TYPE_CMD_COMPLETE:
                        status = msg.get("status", "unknown")
                        if status == "success":
                            return "success"
                        else:
                            error_msg = msg.get("message", "Unknown error")
                            print(f"[Dispatcher] Server error: {error_msg}")
                            return "error"
                
            except queue.Empty:
                # No message yet, continue waiting
                continue
            except Exception as e:
                print(f"[Dispatcher] Error waiting for response: {e}")
                return "error"
        
        # Timeout reached
        print(f"[Dispatcher] Timeout waiting for IK ACK ({self.timeout}s)")
        return "timeout"

