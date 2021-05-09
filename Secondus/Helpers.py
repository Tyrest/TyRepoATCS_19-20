#Helper functions
import numpy as np
import pyscreenshot
import cv2
import mss
from skimage.measure import compare_ssim

monitor = {'top': 40, 'left': 150, 'width': 470, 'height': 410}

def get_screen():
    with mss.mss() as sct:
        screen = np.array(sct.grab(monitor))
        # Simplify image
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.Canny(screen, threshold1 = 200, threshold2=300)
        return screen
'''
# Gets one frame
def get_screen():
    # 470x410 (size of GD without the things
    # behind the cube being recorded and the top being cut off)
    screen = np.array(pyscreenshot.grab(bbox=(150, 40, 620, 450)))
    screen = screen[::2,::2]
    # Simplify image
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.Canny(screen, threshold1 = 200, threshold2=300)
    return screen
'''

def reshape_screen(screen):
    return np.array(screen).reshape(1,410,470,1)

# Compares two following images and returns a boolean for alive. If the image is the "Restart?"
# screen, structural similarity index will be 0.99+ which means the cube is dead. Else, it's alive.
def isalive(screen1, screen2):
    (score, diff) = compare_ssim(screen1, screen2, full=True)
    # print(score)
    if(score < 0.999):
        return True
    else:
        return False

# Records and displays the screen
# def show_screen(title, screen):
#     cv2.imshow(title,screen)
#     #press q to exit screen recording
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
