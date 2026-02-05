import cv2
import os

# Dossier source contenant les images originales
input_folder = "C:/Users/RANA SOUKEUR/Desktop/VIT/DATA"
# Dossier où seront sauvegardées les images augmentées
output_folder = "C:/Users/RANA SOUKEUR/Desktop/NEWDATA"
os.makedirs(output_folder, exist_ok=True)

# Angles de rotation
angles = [-15, -10, -5, 5, 10, 15]

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

# Parcourir toutes les images du dossier
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.tiff', '.tif')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        name, ext = os.path.splitext(filename)
        for angle in angles:
            rotated = rotate_image(image, angle)
            output_path = os.path.join(output_folder, f"{name}_rot{angle}{ext}")
            cv2.imwrite(output_path, rotated)

print("Augmentation terminée.")
