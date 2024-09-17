"""
import cv2
import numpy as np

# Load the potato image
# img = cv2.imread('potato.jpg')

# Malformed Potato
img = cv2.imread('/home/mishkat/Documents/malformed_potato/potato_good_malformed/malformed_potatoes_fourier_2_class/malformed/sd_malformed_12_2_1_16_3.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using   
# Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Apply Fourier transform to the edge image
f = np.fft.fft2(edges)
fshift = np.fft.fftshift(f)

# Create a mask to filter out high-frequency components
mask = np.zeros(fshift.shape, np.uint8)
center_x, center_y = fshift.shape[1] // 2, fshift.shape[0] // 2
radius = 50  # Adjust radius as needed
cv2.circle(mask, (center_x, center_y), radius, 1, thickness=-1)

# Apply the mask to the Fourier transform
fshift_filtered = fshift * mask

# Inverse Fourier transform to get the filtered edge image
ishift = np.fft.ifftshift(fshift_filtered)
filtered_edges = np.fft.ifft2(ishift)
filtered_edges = np.abs(filtered_edges)

# Threshold the filtered edge image to obtain binary edges
thresh = 0.5  # Adjust threshold as needed
binary_edges = np.where(filtered_edges > thresh, 1, 0)

# Find contours in the binary edge image
contours, _ = cv2.findContours(binary_edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Draw contours on the original image
result = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)
cv2.imshow('Filtered Edges', filtered_edges)
binary_edges_uint8 = (binary_edges * 255).astype(np.uint8)  # Normalize and convert to uint8
cv2.imshow('Binary Edges', binary_edges_uint8)
# cv2.imshow('Binary Edges', binary_edges)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

import cv2
import numpy as np

# Load the potato image
img = cv2.imread('/home/mishkat/Documents/malformed_potato/potato_good_malformed/malformed_potatoes_fourier_2_class/malformed/sd_malformed_12_2_1_16_3.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Apply Fourier transform to the edge image
f = np.fft.fft2(edges)
fshift = np.fft.fftshift(f)

# Create a mask to filter out high-frequency components
mask = np.zeros(fshift.shape, np.uint8)
center_x, center_y = fshift.shape[1] // 2, fshift.shape[0] // 2
radius = 50  # Adjust radius as needed
cv2.circle(mask, (center_x, center_y), radius, 1, thickness=-1)

# Apply the mask to the Fourier transform
fshift_filtered = fshift * mask

# Inverse Fourier transform to get the filtered edge image
ishift = np.fft.ifftshift(fshift_filtered)
filtered_edges = np.fft.ifft2(ishift)
filtered_edges = np.abs(filtered_edges)

# Threshold the filtered edge image to obtain binary edges
# thresh = 0.5  # Adjust threshold as needed
# binary_edges = np.where(filtered_edges > thresh, 1, 0)

# Threshold the filtered edge image to obtain binary edges
thresh = 0.3  # Adjust threshold as needed
binary_edges = np.where(filtered_edges > thresh, 1, 0)

# Find contours with hierarchy
contours, hierarchy = cv2.findContours(binary_edges.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter for outermost contours
outer_contours = []
for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == -1:  # Check if parent contour is -1
        outer_contours.append(contour)

# Draw only the outer contours
result = cv2.drawContours(img.copy(), outer_contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)
cv2.imshow('Filtered Edges', filtered_edges)
cv2.imshow('Binary Edges', binary_edges.astype(np.uint8))  # Convert to uint8 for display
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()