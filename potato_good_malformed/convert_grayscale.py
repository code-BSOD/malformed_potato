from PIL import Image
import pathlib

for p in pathlib.Path('/home/mishkat/Downloads/potato_good_malformed/malformed_potatoes_fourier_2_class_gray/malformed').iterdir():
    img = Image.open(p)
    img = img.convert('L')
    img.save(p)