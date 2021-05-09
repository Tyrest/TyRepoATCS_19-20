from pynput.mouse import Button, Controller
import time
from Helpers import get_screen, isalive

mouse = Controller()

def press_LB():
    mouse.position = (180, 350)
    mouse.press(Button.left)

def release_LB():
    mouse.position = (180, 350)
    mouse.release(Button.left)

def restart():
    mouse.position = (180, 425)
    death_screen = get_screen()
    screen = death_screen
    while not isalive(screen, death_screen):
        mouse.click(Button.left, 1)
        screen = get_screen()
    time.sleep(0.25)
