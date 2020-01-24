import gpiozero

### Assign GPIO pins ###
freq = 100
motor1AGpio  = gpiozero.PWMLED(22, frequency = freq)
motor1BGpio  = gpiozero.PWMLED(27, frequency = freq)
motor2AGpio = gpiozero.PWMLED(4 , frequency = freq)
motor2BGpio = gpiozero.PWMLED(17, frequency = freq)

class control:
    turnValue = 0
    speedValue = 0

    @staticmethod
    def turn(turnValue):
        control.turnValue = turnValue

        control.outputGpio()

    @staticmethod
    def speed(speedValue):
        control.speedValue = speedValue

        control.outputGpio()

    @staticmethod
    def outputGpio():
        motor1Value = 0
        motor2Value = 0

        # Find motor speed from turn and speed
        if control.turnValue == 0:
            motor1Value = control.speedValue
            motor2Value = control.speedValue
        else:
            motor1Value = control.speedValue*min(abs(control.turnValue+1), 1)
            motor2Value = control.speedValue*min(abs(control.turnValue-1), 1)

        # Write motor speeds to gpio
        if motor1Value > 0:
            motor1AGpio.value = abs(motor1Value)
            motor1BGpio.value = 0
        else:
            motor1AGpio.value = 0
            motor1BGpio.value = abs(motor1Value)

        if motor2Value > 0:
            motor2AGpio.value = abs(motor2Value)
            motor2BGpio.value = 0
        else:
            motor2AGpio.value = 0
            motor2BGpio.value = abs(motor2Value)

