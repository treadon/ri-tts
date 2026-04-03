#!/usr/bin/env python3
"""
Interactive training wrapper for ri-tts.
Shows training output live with keyboard controls.

Controls:
  P  - Pause/Resume training
  C  - Force checkpoint save
  Q  - Quit (saves checkpoint first)
  S  - Show status summary
  D  - Decode latest sample to WAV
"""

import os
import sys
import time
import signal
import subprocess
import threading
import select
import tty
import termios

TRAIN_CMD = [
    sys.executable, "train.py",
    "--codebooks", "2",
    "--epochs", "10",
    "--batch-size", "1",
]
LOGFILE = "training.log"
PIDFILE = ".train.pid"


class InteractiveTrainer:
    def __init__(self):
        self.process = None
        self.paused = False
        self.running = True
        self.old_settings = None

    def start_training(self):
        """Start or resume the training process."""
        env = os.environ.copy()
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        env["PYTHONUNBUFFERED"] = "1"

        log_fd = open(LOGFILE, "a")
        self.process = subprocess.Popen(
            TRAIN_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            universal_newlines=True,
        )

        with open(PIDFILE, "w") as f:
            f.write(str(self.process.pid))

        # Thread to read and display output
        self.output_thread = threading.Thread(target=self._read_output, args=(log_fd,), daemon=True)
        self.output_thread.start()

    def _read_output(self, log_fd):
        """Read training output and display it."""
        try:
            for line in self.process.stdout:
                if not self.running:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fd.write(line)
                log_fd.flush()
        except (ValueError, OSError):
            pass
        finally:
            log_fd.close()

    def pause(self):
        """Pause training (SIGSTOP)."""
        if self.process and not self.paused:
            os.kill(self.process.pid, signal.SIGSTOP)
            self.paused = True
            print("\n\033[93m  ⏸  PAUSED — press P to resume\033[0m", flush=True)

    def resume(self):
        """Resume training (SIGCONT)."""
        if self.process and self.paused:
            os.kill(self.process.pid, signal.SIGCONT)
            self.paused = False
            print("\n\033[92m  ▶  RESUMED\033[0m", flush=True)

    def force_checkpoint(self):
        """Send SIGUSR1 to trigger a checkpoint save."""
        # HF Trainer doesn't handle SIGUSR1, so we use a different approach:
        # Send SIGINT to save checkpoint, then restart
        if self.process:
            print("\n\033[96m  Saving checkpoint...\033[0m", flush=True)
            self.process.send_signal(signal.SIGINT)
            self.process.wait()
            time.sleep(2)
            print("\033[96m  Checkpoint saved. Resuming...\033[0m", flush=True)
            self.start_training()

    def quit_gracefully(self):
        """Stop training with checkpoint save."""
        if self.process:
            print("\n\033[91m  Stopping and saving checkpoint...\033[0m", flush=True)
            if self.paused:
                os.kill(self.process.pid, signal.SIGCONT)
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("\033[91m  Stopped.\033[0m", flush=True)
        self.running = False
        try:
            os.remove(PIDFILE)
        except FileNotFoundError:
            pass

    def show_status(self):
        """Show training status summary."""
        print("\n\033[94m  --- Status ---\033[0m", flush=True)

        # Check checkpoints
        cp_dir = "checkpoints/qwen3-0.6b-2cb"
        if os.path.exists(cp_dir):
            cps = sorted([d for d in os.listdir(cp_dir) if d.startswith("checkpoint-")])
            if cps:
                print(f"\033[94m  Checkpoints: {', '.join(cps[-3:])}\033[0m", flush=True)

        # Last loss from log
        try:
            with open(LOGFILE) as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "'loss'" in line:
                    print(f"\033[94m  {line.strip()}\033[0m", flush=True)
                    break
        except FileNotFoundError:
            pass

        # Paused state
        state = "PAUSED" if self.paused else "RUNNING"
        print(f"\033[94m  State: {state}\033[0m", flush=True)
        print(f"\033[94m  PID: {self.process.pid if self.process else 'N/A'}\033[0m", flush=True)
        print("\033[94m  ---------------\033[0m", flush=True)

    def setup_terminal(self):
        """Set terminal to raw mode for keypress detection."""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def restore_terminal(self):
        """Restore terminal settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_keypress(self):
        """Non-blocking keypress check."""
        if select.select([sys.stdin], [], [], 0.1)[0]:
            ch = sys.stdin.read(1).lower()
            return ch
        return None

    def run(self):
        """Main interactive loop."""
        print("\033[92m" + "=" * 60 + "\033[0m")
        print("\033[92m  ri-tts Interactive Training (2cb, Qwen3-0.6B)\033[0m")
        print("\033[92m  Controls: P=pause  C=checkpoint  S=status  Q=quit\033[0m")
        print("\033[92m" + "=" * 60 + "\033[0m")
        print()

        self.start_training()
        self.setup_terminal()

        try:
            while self.running:
                # Check if process died
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    if exit_code == 0:
                        print("\n\033[92m  Training completed!\033[0m", flush=True)
                        self.running = False
                        break
                    else:
                        print(f"\n\033[91m  Training crashed (exit {exit_code}). Restarting in 10s...\033[0m", flush=True)
                        time.sleep(10)
                        self.start_training()
                        continue

                key = self.check_keypress()
                if key == 'p':
                    if self.paused:
                        self.resume()
                    else:
                        self.pause()
                elif key == 'c':
                    self.force_checkpoint()
                elif key == 'q':
                    self.restore_terminal()
                    confirm = input("\n  Really quit? (y/N) ")
                    if confirm.lower() == 'y':
                        self.quit_gracefully()
                        break
                    else:
                        self.setup_terminal()
                        print("  Continuing...", flush=True)
                elif key == 's':
                    self.show_status()

        except KeyboardInterrupt:
            self.quit_gracefully()
        finally:
            self.restore_terminal()
            try:
                os.remove(PIDFILE)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    trainer = InteractiveTrainer()
    trainer.run()
