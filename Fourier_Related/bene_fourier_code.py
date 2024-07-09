import cv2
import numpy as np
from scipy.interpolate import interp1d
from pyefd import elliptic_fourier_descriptors

def process_image_and_calculate_dft(image_path, target_length=256):

    img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED )
    cv2.imshow('blur', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    cv2.imshow('blur',gray)
    cv2.imshow('contour', thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    coeffs = []
    for cnt in contours:
        # Find the coefficients of all contours
        coeffs.append(elliptic_fourier_descriptors(
            np.squeeze(cnt), order=10))
        


    potato_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(potato_contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    potato_contour = potato_contour.squeeze()
    distances = np.sqrt((potato_contour[:, 0] - cx) ** 2 + (potato_contour[:, 1] - cy) ** 2)
    
    x_old = np.linspace(0, 1, len(distances))
    f = interp1d(x_old, distances, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    distances_interpolated = f(x_new)

    # Normalize the interpolated distances so that the maximum distance is 250
    max_distance = np.max(distances_interpolated)
    distances_normalized = (distances_interpolated / max_distance) * 250

    dft_result = np.fft(distances_normalized)
    #dft_result = fft(distances_interpolated)
    dft_magnitude = np.abs(dft_result)
    return dft_magnitude

folder_path = r"C:\Users\admin\Karevo\Karevo - Dokumente\01_Technik\Images-Videos\ProcessedData\240206_Segmented_Malformed\Flaschenhals"
# Find the contours of a binary image using OpenCV.
contours, hierarchy = cv2.findContours(
    im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through all contours found and store each contour's
# elliptical Fourier descriptor's coefficients.
