import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from pyefd import elliptic_fourier_descriptors
from sklearn.cluster import KMeans
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

def read_images_and_extract_contours_single_image(img_location):
    """Read images from a folder and extract their largest external contour."""
    images = img_location
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


def predict_cluster(image_path, kmeans, order=30):
    """Predicts the cluster for a single image and returns its label and average shape."""
    contour = read_images_and_extract_contours_single_image([image_path])[0]
    fourier_coeffs = compute_fourier_coefficients(contour, order=order).flatten()
    cluster_label = kmeans.predict([fourier_coeffs])[0]
    class_averages = compute_class_averages([fourier_coeffs], [cluster_label], kmeans.n_clusters)
    avg_shape = inverse_fourier_transform(class_averages[cluster_label].reshape(order, -1), H=order)
    return cluster_label, avg_shape





def old_main(folder_path, n_clusters=36, order=30):
    contours = read_images_and_extract_contours(folder_path)
    # Compute Fourier coefficients and flatten them into a 1D array per contour
    fourier_descriptors = [compute_fourier_coefficients(contour, order=order).flatten() for contour in contours]

    # Convert list of 1D arrays into a 2D array for KMeans
    fourier_descriptors = np.array(fourier_descriptors)

    if fourier_descriptors.shape[0] > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fourier_descriptors)
        labels = kmeans.labels_
        class_averages = compute_class_averages(fourier_descriptors, labels, n_clusters)

        # Determine subplot grid size
        cols = min(n_clusters, 12)
        rows = n_clusters // cols + (n_clusters % cols > 0)

        # Plot individual clusters in subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.75))
        axes = axes.flatten() if n_clusters > 1 else [axes]

        for i, avg_coeffs in enumerate(class_averages):
            if avg_coeffs is not None:
                shape_model = inverse_fourier_transform(avg_coeffs.reshape(order, -1), H=order)
                ax = axes[i]
                ax.plot(shape_model[:, 0], shape_model[:, 1], label=f'Cluster {i + 1}')
                ax.set_title(f'Cluster {i + 1}')
                ax.axis('equal')
                ax.legend()

        # Adjust layout
        plt.tight_layout()
        plt.show()   
        
    else:
        print("Not enough contours for KMeans clustering.")

def plot_predicted_clusters(predicted_cluster, avg_shape, image_to_predict_path):
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the image
    img = cv2.imread(image_to_predict_path)
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Input Image')
    axes[0].axis('off')  # Turn off axis for the image
    
    #Show the average shape of the associated cluster
    axes[1].plot(avg_shape[:, 0], avg_shape[:, 1])
    axes[1].set_title(f'Predicted Cluster: {predicted_cluster + 1}')
    axes[1].axis('equal')
    axes[1].set_aspect('equal')

    plt.show()
    # plt.pause(2)
    # plt.waitforbuttonpress()
    # plt.close(fig)
    # plt.pause(0.1)
    

    # fig.savefig(f'predicted_cluster_{image_to_predict_path}.png')
    # fig.savefig()


def main(folder_path, n_clusters=36, order=30):
    contours = read_images_and_extract_contours(folder_path)
    fourier_descriptors = np.array([compute_fourier_coefficients(c, order=order).flatten() for c in contours])

    if fourier_descriptors.shape[0] > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fourier_descriptors)
        labels = kmeans.labels_
        class_averages = compute_class_averages(fourier_descriptors, labels, n_clusters)
        
        # Determine subplot grid size
        # cols = min(n_clusters, 12)
        # rows = n_clusters // cols + (n_clusters % cols > 0)

        # # Plot individual clusters in subplots
        # fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.75))
        # axes = axes.flatten() if n_clusters > 1 else [axes]

        # for i, avg_coeffs in enumerate(class_averages):
        #     if avg_coeffs is not None:
        #         shape_model = inverse_fourier_transform(avg_coeffs.reshape(order, -1), H=order)
        #         ax = axes[i]
        #         ax.plot(shape_model[:, 0], shape_model[:, 1], label=f'Cluster {i + 1}')
        #         ax.set_title(f'Cluster {i + 1}')
        #         ax.axis('equal')
        #         ax.legend()

        # # Adjust layout
        # plt.tight_layout()
        # plt.show()

        # Input the image to predict
        

        while True:
            image_to_predict_path = input("Enter the path to the image you want to predict (or type 'q' to quit): ")
            if image_to_predict_path.lower() == 'q':
                break

            predicted_cluster, avg_shape = predict_cluster(image_to_predict_path, kmeans, order=order)

            # Plot the predicted cluster
            plt.ion()
            plot_predicted_clusters(predicted_cluster, avg_shape, image_to_predict_path)            

    else:
        print("Not enough contours for KMeans clustering.")

if __name__ == "__main__":
    folder_path = r"/home/mishkat/Documents/malformed"
    # start = time.time()
    main(folder_path)
    end = time.time()  
    # print(f"Time taken: {end - start:.2f} seconds")