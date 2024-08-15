""" Okretanje i zakretanje slika """

#!/usr/bin/python
import os
from PIL import Image

path = "Slike/"
dirs = os.listdir(path)


# okreni vodoravno


def flip_me():
    # vodoravno okreće sve slike u mapi
    counter = 1
    for item in dirs:
        im = Image.open(path + item)
        horz_img = im.transpose(Image.FLIP_LEFT_RIGHT)
        horz_img.save("h_mala_" + str(counter) + " _128.jpg", "JPEG", quality=90)
        counter += 1


# okreni okomito
# vertical_img = original_img.transpose(method=Image.FLIP_TOP_BOTTOM)

flip_me()
