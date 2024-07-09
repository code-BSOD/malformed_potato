import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from pyefd import elliptic_fourier_descriptors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time


def read_images_and_extract_contours(folder_path):
    """Read images from a folder and extract their largest external contour."""
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    contours_list = []
    for image_path in images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour = contour.squeeze()
            contours_list.append(normalize_contour_points(contour))
    return contours_list

def normalize_contour_points(contour):
    """Normalize contour points to a fixed number for consistency."""
    N = 250
    contour_length = np.linspace(0, 1, len(contour))
    normalized_length = np.linspace(0, 1, N)
    interp_func_x = interp1d(contour_length, contour[:, 0], kind='linear')
    interp_func_y = interp1d(contour_length, contour[:, 1], kind='linear')
    normalized_contour = np.vstack((interp_func_x(normalized_length), interp_func_y(normalized_length))).T
    return normalized_contour

def compute_fourier_coefficients(contour, order=30):
    """Compute Fourier coefficients for a given contour."""
    coeffs = elliptic_fourier_descriptors(contour, order=order, normalize=True)
    #return coeffs[1:]  # Skip the first coefficient as it's related to the image position
    return coeffs
def compute_class_averages(fourier_descriptors, labels, n_clusters):
    """Compute average Fourier coefficients for each cluster."""
    sums = [np.zeros(fourier_descriptors[0].shape) for _ in range(n_clusters)]
    counts = [0] * n_clusters
    for coeffs, label in zip(fourier_descriptors, labels):
        sums[label] += coeffs
        counts[label] += 1
    averages = [sums[i] / counts[i] if counts[i] > 0 else None for i in range(n_clusters)]
    return averages

def inverse_fourier_transform(coeffs, num_points=500, H=30):
    """Reconstruct a shape from its Fourier coefficients."""
    X, Y = np.zeros(num_points), np.zeros(num_points)
    t_values = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    for h in range(H):
        X += coeffs[h, 0] * np.cos((h+1) * t_values) + coeffs[h, 1] * np.sin((h+1) * t_values)
        Y += coeffs[h, 2] * np.cos((h+1) * t_values) + coeffs[h, 3] * np.sin((h+1) * t_values)
    return np.vstack((X, Y)).T


#def main(folder_path, n_clusters=8, order=30):
    contours = read_images_and_extract_contours(folder_path)
    # Compute Fourier coefficients and flatten them into a 1D array per contour
    fourier_descriptors = [compute_fourier_coefficients(contour, order=order).flatten() for contour in contours]

    # Convert list of 1D arrays into a 2D array for KMeans
    fourier_descriptors = np.array(fourier_descriptors)

    # Ensure we have more than one descriptor to fit KMeans
    if fourier_descriptors.shape[0] > 1:
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fourier_descriptors)
        dbscans = DBSCAN(eps=0.5, min_samples=5).fit(fourier_descriptors)
        # class_averages = compute_class_averages(fourier_descriptors, kmeans.labels_, n_clusters)
        class_averages = compute_class_averages(fourier_descriptors, dbscans.labels_, len(set(dbscans.labels_)))

        # Plot average shapes
        for i, avg_coeffs in enumerate(class_averages):
            if avg_coeffs is not None:
                shape_model = inverse_fourier_transform(avg_coeffs.reshape(order, -1), H=order)
                plt.plot(shape_model[:, 0], shape_model[:, 1], label=f'Cluster {i + 1}')
        plt.axis('equal')
        plt.legend()
        plt.title('Average Shape Models for Each Cluster')
        plt.show()
    else:
        print("Not enough contours for KMeans clustering.")



def main(folder_path, eps=0.5, min_samples=5, order=30):
    contours = read_images_and_extract_contours(folder_path)
    fourier_descriptors = np.array([compute_fourier_coefficients(c, order).flatten() for c in contours])

    if fourier_descriptors.shape[0] > 1:
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(fourier_descriptors)
        labels = dbscan.labels_

        # Identify unique cluster labels (excluding noise, if present)
        unique_labels = set(labels) - {-1}  # Exclude noise label -1

        # Plot individual clusters
        cols = min(len(unique_labels), 12)
        rows = len(unique_labels) // cols + (len(unique_labels) % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.75))
        axes = axes.flatten() if len(unique_labels) > 1 else [axes]

        for i, label in enumerate(unique_labels):
            cluster_mask = (labels == label)
            cluster_descriptors = fourier_descriptors[cluster_mask]
            if cluster_descriptors.size > 0:
                avg_coeffs = cluster_descriptors.mean(axis=0)
                shape_model = inverse_fourier_transform(avg_coeffs.reshape(order, -1), H=order)
                ax = axes[i]
                ax.plot(shape_model[:, 0], shape_model[:, 1])
                ax.set_title(f'Cluster {label + 1}')
                ax.axis('equal')

        # Adjust layout
        plt.tight_layout()
        plt.show()

        # Plot all clusters together (including noise)
        plt.figure(figsize=(8, 6))
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels) + 1))  # Add color for noise
        for k, col in zip(unique_labels | {-1}, colors): # Include noise label
            class_member_mask = (labels == k)
            xy = fourier_descriptors[class_member_mask]
            for coeffs in xy:
                shape_model = inverse_fourier_transform(coeffs.reshape(order, -1), H=order)
                plt.plot(shape_model[:, 0], shape_model[:, 1], 'o', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6, label=f'Cluster {k + 1}' if k != -1 else 'Noise')

        plt.title('All Clusters (DBSCAN)')
        plt.legend()
        plt.show()
    else:
        print("Not enough contours for DBSCAN clustering.")


if __name__ == "__main__":
    folder_path = r"/home/mishkat/Documents/malformed"
    start = time.time()
    main(folder_path)
    end = time.time()  
    print(f"Time taken: {end - start:.2f} seconds")