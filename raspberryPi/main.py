from classes.control import control
import socketio
import time

sio = socketio.Client()

last_packet_time = 0

### Settings ###
center_offset_threshold = 0
speed_multiplier = 0.2
max_speed = 0.1
turn_multiplier = 2

@sio.event
def connect():
    sio.emit("new_frame")
    print('connection established')

@sio.event
def disconnect():
    print('disconnected from server')

@sio.on("driving_data")
def handle_data(data):
    global last_packet_time
    
    last_packet_time = time.time()

    speed = data["speed"]
    speed = max(0, min(1-speed*speed_multiplier, max_speed))
    control.speed(speed)

    center_offset = data["center_offset"]
    turn_value = max(-1, min(center_offset*turn_multiplier, 1))
    print(turn_value)

    control.turn(turn_value)

    ### Request new frame ###
    sio.emit("new_frame")

#sio.connect('http://192.168.1.180:8081')
sio.connect('http://192.168.1.248:8080')

while True:
    if time.time() - last_packet_time > 1:
        control.speed(0)
    pass
max_speed = 0.6
sio.wait()
