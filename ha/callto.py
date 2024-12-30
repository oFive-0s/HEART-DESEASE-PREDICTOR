import socket
import threading
import pyaudio
import time

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Network settings
LOCAL_IP = '192.168.4.7'  # Change this to '192.168.1.10' for Windows   192.168.4.7
REMOTE_IP = '192.168.137.57'  # Change this to '192.168.1.9' for Windows
PORT = 12345

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_IP, PORT))

# Audio stream for recording
stream_in = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Audio stream for playback
stream_out = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# Buffer for incoming audio
audio_buffer = []
BUFFER_SIZE = 10  # Number of chunks to buffer

# Lock for thread-safe buffer access
buffer_lock = threading.Lock()

def send_audio():
    while True:
        try:
            data = stream_in.read(CHUNK)
            sock.sendto(data, (REMOTE_IP, PORT))
        except Exception as e:
            print(f"Error sending audio: {e}")

def receive_audio():
    global audio_buffer
    while True:
        try:
            data, addr = sock.recvfrom(CHUNK * 2)
            with buffer_lock:
                audio_buffer.append(data)
                if len(audio_buffer) > BUFFER_SIZE:
                    audio_buffer.pop(0)
        except Exception as e:
            print(f"Error receiving audio: {e}")

def play_audio():
    global audio_buffer
    while True:
        if audio_buffer:
            with buffer_lock:
                data = audio_buffer.pop(0)
            stream_out.write(data)
        else:
            time.sleep(0.001)  # Small delay to prevent busy-waiting

# Start threads
threading.Thread(target=send_audio, daemon=True).start()
threading.Thread(target=receive_audio, daemon=True).start()
threading.Thread(target=play_audio, daemon=True).start()

print("Voice communication system is running. Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")

# Clean up
stream_in.stop_stream()
stream_in.close()
stream_out.stop_stream()
stream_out.close()
audio.terminate()
sock.close()