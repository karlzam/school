from PIL import Image
import os.path, sys

path = r"C:\Users\kzammit\Repos\school\phys555\project-DL\orcas_classification\cropped_spectrograms"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            # last one does the height
            # left top right bottom
            imCrop = im.crop((80, 58, 475, 425)) #corrected
            imCrop.save(f + '-crop.png', "PNG", quality=300)


crop()