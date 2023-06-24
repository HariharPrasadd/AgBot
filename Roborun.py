import serial
import time
import curses

screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

port = '/dev/ttyACM0'
baud = 9600

ser = serial.Serial(port, baud, timeout = 1)

time.sleep(2) 

try:
	while True:
		time.sleep(0.015)
		keypress = screen.getch()
		if keypress == ord('q'):
			break
		elif keypress == curses.KEY_UP:
			ser.write(b'W')
		elif keypress == curses.KEY_DOWN:
			ser.write(b'S')
		elif keypress == curses.KEY_RIGHT:
			ser.write(b'D')
		elif keypress == curses.KEY_LEFT:
			ser.write(b'A')
		elif keypress == ' ':
			ser.write(b'Q')
finally:
	curses.nocbreak()
	screen.keypad(False)
	curses.echo()
	curses.endwin()
	