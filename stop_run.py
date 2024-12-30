import argparse
import socket

def send_stop_signal(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', port))
        s.sendall(b'STOP')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop_run', action='store_true', help='Send stop signal to stop_run')
    parser.add_argument('--stop_runs', action='store_true', help='Send stop signal to stop_runs')
    args = parser.parse_args()

    if args.stop_run:
        # Send stop signal to stop_run
        send_stop_signal(65432)
        print('Stop signal sent to stop_run')

    if args.stop_runs:
        # Send stop signal to stop_runs
        send_stop_signal(65433)
        print('Stop signal sent to stop_runs')

if __name__ == '__main__':
    main()