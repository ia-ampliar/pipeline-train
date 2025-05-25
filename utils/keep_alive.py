import subprocess
import time

def keep_alive(interval=300):
    """Executa um comando simples no terminal a cada 'interval' segundos."""
    while True:
        subprocess.run(["echo", "keep alive"])
        time.sleep(interval)

if __name__ == "__main__":
    keep_alive(600)  # (1200 segundos = 20 minutos)
