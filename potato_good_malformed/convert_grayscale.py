from PIL import Image
import pathlib

for p in pathlib.Path('/home/mishkat/Downloads/malformed_potato/potato_good_malformed/malformed_potatoes_fourier_3_class_gray/ugly').iterdir():
    img = Image.open(p)
    img = img.convert('L')
    img.save(p)