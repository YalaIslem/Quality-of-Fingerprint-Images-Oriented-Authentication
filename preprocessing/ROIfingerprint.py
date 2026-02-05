import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_fingerprint_roi(image_path, visualize=True):
    # 1. Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Appliquer un seuillage adaptatif pour isoler les crêtes
    thresh = cv2.adaptiveThreshold(image, 255, 
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # 3. Morphologie pour remplir les trous et réduire le bruit
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Trouver les contours pour délimiter la zone utile
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Trouver le plus grand contour (probable empreinte)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 6. Extraire la ROI
    roi = image[y:y+h, x:x+w]

    if visualize:
        # Afficher les résultats
        cv2.rectangle(image, (x, y), (x+w, y+h), 255, 2)
        plt.subplot(1, 2, 1)
        plt.title("Détection de la ROI")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("ROI extraite")
        plt.imshow(roi, cmap='gray')
        plt.show()

    return roi

# Exemple d'utilisation
roi = extract_fingerprint_roi("101_2.tif")
