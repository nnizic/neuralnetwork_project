""" mijenjanje veličine slika """

#!/usr/bin/python
import os
import sys
from PIL import Image

path = "Neskalirane/"
dirs = os.listdir(path)


def resize():
    # mjenja veličinu svim slikama u mapi
    counter = 1
    for item in dirs:
        im = Image.open(path + item)

        # izreži sliku tako da uzmeš sredinu slike
        wdiff = (im.size[0] - min(im.size)) // 2
        hdiff = (im.size[1] - min(im.size)) // 2
        im = im.crop((wdiff, hdiff, im.size[0] - wdiff, im.size[1] - hdiff))

        # imResize = im.resize((3072, 3072), Image.Resampling.LANCZOS) # ovo je za provjeru ako je rastegnuta slika
        imResize = im.resize((128, 128), Image.Resampling.LANCZOS)
        imResize.save("mala_" + str(counter) + "_128.jpg", "JPEG", quality=90)
        counter += 1


resize()
