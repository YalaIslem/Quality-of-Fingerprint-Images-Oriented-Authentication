# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 13:50:30 2025

@author: DELL
"""

import cv2
import os
import math
import re
from scipy.optimize import linear_sum_assignment
import shutil
import fingerprint_feature_extractor

# --- Fonction pour tri naturel des fichiers ---
def natural_key(s):
    """Utilise des entiers pour trier naturellement les noms de fichiers comme 1_1.bmp, 2_1.bmp, 10_1.bmp..."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# --- Extraction séparée des terminaisons et bifurcations ---
def extract_features(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")  
    
    term, bif = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=False
    )
    
    term_list = []
    bif_list = []
    
    for m in term:
        orientation = m.Orientation[0] if isinstance(m.Orientation, list) else m.Orientation
        term_list.append((m.locX, m.locY, orientation))
    
    for m in bif:
        orientation = m.Orientation[0] if isinstance(m.Orientation, list) else m.Orientation
        bif_list.append((m.locX, m.locY, orientation))

    return term_list, bif_list

# --- Centrage des minuties ---
def center_minutiae(minutiae):
    if not minutiae:
        return []
    cx = sum(m[0] for m in minutiae) / len(minutiae)
    cy = sum(m[1] for m in minutiae) / len(minutiae)
    return [(m[0] - cx, m[1] - cy, m[2]) for m in minutiae]

# --- Distance combinée position + angle ---
def combined_distance(m1, m2, angle_weight=0.5):
    dx = m1[0] - m2[0]
    dy = m1[1] - m2[1]
    angle_diff = abs(m1[2] - m2[2]) % 180
    angle_diff = min(angle_diff, 180 - angle_diff)
    return math.sqrt(dx**2 + dy**2) + angle_weight * angle_diff

# --- Matching optimal (même type uniquement) ---
def match_minutiae_optimally(list_a, list_b, distance_thresh=70, angle_weight=0.5):
    if not list_a or not list_b:
        return 0

    list_a = center_minutiae(list_a)
    list_b = center_minutiae(list_b)

    cost_matrix = []
    for m1 in list_a:
        row = [combined_distance(m1, m2, angle_weight) for m2 in list_b]
        cost_matrix.append(row)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched = 0
    for i, j in zip(row_ind, col_ind):
        if combined_distance(list_a[i], list_b[j], angle_weight) <= distance_thresh:
            matched += 1
    return matched

# --- Score de similarité basé sur bif↔bif et term↔term ---
def calculate_similarity_score(ref_term, ref_bif, tgt_term, tgt_bif):
    matched_term = match_minutiae_optimally(ref_term, tgt_term)
    matched_bif = match_minutiae_optimally(ref_bif, tgt_bif)

    total_matched = matched_term + matched_bif
    total_ref = len(ref_term) + len(ref_bif)
    total_tgt = len(tgt_term) + len(tgt_bif)

    if total_ref == 0 or total_tgt == 0:
        return 0, total_matched
    score = (total_matched / ((total_ref + total_tgt) / 2)) * 100
    # score = (total_matched / max(total_ref, total_tgt)) * 100
    return score, total_matched

# --- Classification par score ---
def classify_image(score):
    if score > 80:
        return "Class 5"
    elif score > 70:
        return "Class 4"
    elif score > 60:
        return "Class 3"
    elif score > 50:
        return "Class 2"
    else:
        return "Class 1"

# --- Comparaison des empreintes d'une même personne ---
def compare_all_fingerprints_for_person(person_id, folder_path, images):
    print(f"\n--- Comparaison des empreintes pour la personne {person_id} ---")
    person_images = sorted(
        [img for img in images if img.startswith(f"{person_id}_")],
        key=natural_key
    )
    
    extracted = {}
    for img_name in person_images:
        path = os.path.join(folder_path, img_name)
        try:
            extracted[img_name] = extract_features(path)
        except Exception as e:
            print(f"Erreur d'extraction {img_name} : {e}")
    
    # Calcul de la moyenne des scores pour chaque image de référence
    avg_scores = {}
    for ref_name in person_images:
        ref_feats = extracted.get(ref_name)
        if not ref_feats:
            continue
        ref_term, ref_bif = ref_feats
        total_score = 0
        total_comparisons = 0
        
        for compare_name in person_images:
            if compare_name == ref_name:
                continue
            comp_feats = extracted.get(compare_name)
            if not comp_feats:
                continue
            comp_term, comp_bif = comp_feats
            score, _ = calculate_similarity_score(ref_term, ref_bif, comp_term, comp_bif)
            total_score += score
            total_comparisons += 1
        
        avg_scores[ref_name] = total_score / total_comparisons if total_comparisons > 0 else 0
    
    # Choisir l'image de référence avec la meilleure moyenne
    best_ref_name = max(avg_scores, key=avg_scores.get)
    print(f"Meilleure image de référence pour {person_id}: {best_ref_name} avec une moyenne de {avg_scores[best_ref_name]:.2f}%")

    # Comparaison avec la meilleure image de référence
    best_ref_feats = extracted.get(best_ref_name)
    if not best_ref_feats:
        return
    
    best_ref_term, best_ref_bif = best_ref_feats
    for compare_name in person_images:
        comp_feats = extracted.get(compare_name)
        if not comp_feats:
            continue
        comp_term, comp_bif = comp_feats
        score, matched = calculate_similarity_score(best_ref_term, best_ref_bif, comp_term, comp_bif)
        category = classify_image(score)
        print(f"{compare_name} → Score: {score:.2f}% → {category} → Minuties appariées: {matched}")
        
        # Copier les images dans les bons dossiers
        class_folder = f"{category.replace(' ', '_')}"
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        shutil.copy(os.path.join(folder_path, compare_name), os.path.join(class_folder, compare_name))
# --- Comparaison avec une seule image de référence ---
def compare_with_reference(reference_image, folder_path, images):
    print(f"\n--- Comparaison avec l'empreinte de référence {reference_image} ---")
    ref_path = os.path.join(folder_path, reference_image)
    try:
        ref_term, ref_bif = extract_features(ref_path)
    except Exception as e:
        print(f"Erreur d’extraction pour {reference_image} : {e}")
        return

    for img_name in images:
    
        img_path = os.path.join(folder_path, img_name)
        try:
            img_term, img_bif = extract_features(img_path)
        except Exception as e:
            print(f"Erreur d’extraction pour {img_name} : {e}")
            continue
        score, matched = calculate_similarity_score(ref_term, ref_bif, img_term, img_bif)
        category = classify_image(score)
        print(f"{img_name} → Score: {score:.2f}% → {category} → Minuties appariées: {matched} ")
        
        # (Ref: {len(ref_term)+len(ref_bif)}, Cible: {len(img_term)+len(img_bif)})
# === Exécution principale ===
if __name__ == "__main__":
    folder = "C:/Users/DELL/Desktop/DB3_B"
    all_images = sorted(
        [f for f in os.listdir(folder) if f.endswith(".tif")],
        key=natural_key
    )
    person_ids = sorted(set(f.split("_")[0] for f in all_images))

    print("Choisissez une option :")
    print("1 - Comparaison entre les empreintes d'une même personne")
    print("2 - Comparaison d'une empreinte de référence avec toutes les autres")
    choice = input("Votre choix (1 ou 2) : ")

    
    

    # Comparer les empreintes pour chaque personne
    
    if choice == "1":
        for person_id in person_ids:
            compare_all_fingerprints_for_person(person_id, folder, all_images)
    elif choice == "2":
        reference_image = input("Entrez le nom de l'empreinte de référence (ex: 101_1.tif): ")
        if reference_image not in all_images:
            print(f"L'empreinte {reference_image} n'existe pas dans le dossier.")
        else:
            compare_with_reference(reference_image, folder, all_images)
    else:
        print("Choix invalide.")
       
       
       
       
# # -*- coding: utf-8 -*-
# """
# Created on Sun Apr 27 13:50:30 2025

# @author: DELL
# """

# import cv2
# import os
# import math
# import re
# import numpy as np
# from scipy.optimize import linear_sum_assignment
# import shutil
# import fingerprint_feature_extractor

# # --- Fonction pour tri naturel des fichiers ---
# def natural_key(s):
#     """Utilise des entiers pour trier naturellement les noms de fichiers comme 1_1.bmp, 2_1.bmp, 10_1.bmp..."""
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# # --- Extraction séparée des terminaisons et bifurcations ---
# def extract_features(image_path):
#     img = cv2.imread(image_path, 0)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")  
    
#     # Calcul du contraste (écart-type)
#     contrast = np.std(img)
    
#     # Extraction des minuties
#     term, bif = fingerprint_feature_extractor.extract_minutiae_features(
#         img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False
#     )
    
#     term_list = []
#     bif_list = []
    
#     for m in term:
#         orientation = m.Orientation[0] if isinstance(m.Orientation, list) else m.Orientation
#         term_list.append((m.locX, m.locY, orientation))
    
#     for m in bif:
#         orientation = m.Orientation[0] if isinstance(m.Orientation, list) else m.Orientation
#         bif_list.append((m.locX, m.locY, orientation))

#     return term_list, bif_list, contrast, len(term_list) + len(bif_list)

# # --- Centrage des minuties ---
# def center_minutiae(minutiae):
#     if not minutiae:
#         return []
#     cx = sum(m[0] for m in minutiae) / len(minutiae)
#     cy = sum(m[1] for m in minutiae) / len(minutiae)
#     return [(m[0] - cx, m[1] - cy, m[2]) for m in minutiae]

# # --- Distance combinée position + angle ---
# def combined_distance(m1, m2, angle_weight=0.5):
#     dx = m1[0] - m2[0]
#     dy = m1[1] - m2[1]
#     angle_diff = abs(m1[2] - m2[2]) % 180
#     angle_diff = min(angle_diff, 180 - angle_diff)
#     return math.sqrt(dx**2 + dy**2) + angle_weight * angle_diff

# # --- Matching optimal (même type uniquement) ---
# def match_minutiae_optimally(list_a, list_b, distance_thresh=30, angle_weight=0.5):
#     if not list_a or not list_b:
#         return 0

#     list_a = center_minutiae(list_a)
#     list_b = center_minutiae(list_b)

#     cost_matrix = []
#     for m1 in list_a:
#         row = [combined_distance(m1, m2, angle_weight) for m2 in list_b]
#         cost_matrix.append(row)

#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     matched = 0
#     for i, j in zip(row_ind, col_ind):
#         if combined_distance(list_a[i], list_b[j], angle_weight) <= distance_thresh:
#             matched += 1
#     return matched

# # --- Score de similarité basé sur bif↔bif et term↔term ---
# def calculate_similarity_score(ref_term, ref_bif, tgt_term, tgt_bif):
#     matched_term = match_minutiae_optimally(ref_term, tgt_term)
#     matched_bif = match_minutiae_optimally(ref_bif, tgt_bif)

#     total_matched = matched_term + matched_bif
#     total_ref = len(ref_term) + len(ref_bif)
#     total_tgt = len(tgt_term) + len(tgt_bif)

#     if total_ref == 0 or total_tgt == 0:
#         return 0, total_matched, total_ref, total_tgt
#     # score = (total_matched / ((total_ref + total_tgt) / 2)) * 100

#     score = (total_matched / max(total_ref, total_tgt)) * 100
#     return score, total_matched, total_ref, total_tgt

# # --- Classification par score ---
# def classify_image(score):
#     if score > 80:
#         return "Class 5"
#     elif score > 70:
#         return "Class 4"
#     elif score > 60:
#         return "Class 3"
#     elif score > 50:
#         return "Class 2"
#     else:
#         return "Class 1"

# # --- Comparaison des empreintes d'une même personne ---
# def compare_all_fingerprints_for_person(person_id, folder_path, images):
#     print(f"\n--- Comparaison des empreintes pour la personne {person_id} ---")
#     person_images = sorted(
#         [img for img in images if img.startswith(f"{person_id}_")],
#         key=natural_key
#     )
    
#     extracted = {}
#     for img_name in person_images:
#         path = os.path.join(folder_path, img_name)
#         try:
#             term, bif, contrast, minutiae_count = extract_features(path)
#             extracted[img_name] = (term, bif, contrast, minutiae_count)
#         except Exception as e:
#             print(f"Erreur d'extraction {img_name} : {e}")
    
#     # Calcul de la moyenne des scores pour chaque image de référence
#     avg_scores = {}
#     for ref_name in person_images:
#         ref_feats = extracted.get(ref_name)
#         if not ref_feats:
#             continue
#         ref_term, ref_bif, _, _ = ref_feats
#         total_score = 0
#         total_comparisons = 0
        
#         for compare_name in person_images:
#             if compare_name == ref_name:
#                 continue
#             comp_feats = extracted.get(compare_name)
#             if not comp_feats:
#                 continue
#             comp_term, comp_bif, _, _ = comp_feats
#             score, _, _, _ = calculate_similarity_score(ref_term, ref_bif, comp_term, comp_bif)
#             total_score += score
#             total_comparisons += 1
        
#         avg_scores[ref_name] = total_score / total_comparisons if total_comparisons > 0 else 0
    
#     # Choisir l'image de référence avec la meilleure moyenne
#     best_ref_name = max(avg_scores, key=avg_scores.get)
#     print(f"Meilleure image de référence pour {person_id}: {best_ref_name} avec une moyenne de {avg_scores[best_ref_name]:.2f}%")

#     # Comparaison avec la meilleure image de référence
#     best_ref_feats = extracted.get(best_ref_name)
#     if not best_ref_feats:
#         return
    
#     best_ref_term, best_ref_bif, ref_contrast, ref_minutiae = best_ref_feats
#     for compare_name in person_images:
#         comp_feats = extracted.get(compare_name)
#         if not comp_feats:
#             continue
#         comp_term, comp_bif, comp_contrast, comp_minutiae = comp_feats
#         score, matched, ref_total, comp_total = calculate_similarity_score(best_ref_term, best_ref_bif, comp_term, comp_bif)
#         category = classify_image(score)
#         print(f"{compare_name} → Score: {score:.2f}% → {category}")
#         print(f"  Minuties appariées: {matched} | Ref: {ref_total} | Test: {comp_total}")
#         print(f"  Contraste - Ref: {ref_contrast:.1f} | Test: {comp_contrast:.1f}")
        
#         # Copier les images dans les bons dossiers
#         class_folder = f"{category.replace(' ', '_')}"
#         if not os.path.exists(class_folder):
#             os.makedirs(class_folder)
#         shutil.copy(os.path.join(folder_path, compare_name), os.path.join(class_folder, compare_name))

# # --- Comparaison avec une seule image de référence ---
# def compare_with_reference(reference_image, folder_path, images):
#     print(f"\n--- Comparaison avec l'empreinte de référence {reference_image} ---")
#     ref_path = os.path.join(folder_path, reference_image)
#     try:
#         ref_term, ref_bif, ref_contrast, ref_minutiae = extract_features(ref_path)
#     except Exception as e:
#         print(f"Erreur d'extraction pour {reference_image} : {e}")
#         return

#     for img_name in images:
#         if img_name == reference_image:
#             continue
            
#         img_path = os.path.join(folder_path, img_name)
#         try:
#             img_term, img_bif, img_contrast, img_minutiae = extract_features(img_path)
#         except Exception as e:
#             print(f"Erreur d'extraction pour {img_name} : {e}")
#             continue
            
#         score, matched, ref_total, img_total = calculate_similarity_score(ref_term, ref_bif, img_term, img_bif)
#         category = classify_image(score)
        
#         print(f"\n{img_name} → Score: {score:.2f}% → {category}")
#         print(f"  Minuties appariées: {matched} | Ref: {ref_total} | Test: {img_total}")
#         print(f"  Contraste - Ref: {ref_contrast:.1f} | Test: {img_contrast:.1f}")

# # === Exécution principale ===
# if __name__ == "__main__":
#     folder = r"C:\Users\DELL\Desktop\DB4_B_ROI"
#     all_images = sorted(
#         [f for f in os.listdir(folder) if f.endswith(".tif")],
#         key=natural_key
#     )
#     person_ids = sorted(set(f.split("_")[0] for f in all_images))

#     print("Choisissez une option :")
#     print("1 - Comparaison entre les empreintes d'une même personne")
#     print("2 - Comparaison d'une empreinte de référence avec toutes les autres")
#     choice = input("Votre choix (1 ou 2) : ")

#     if choice == "1":
#         for person_id in person_ids:
#             compare_all_fingerprints_for_person(person_id, folder, all_images)
#     elif choice == "2":
#         reference_image = input("Entrez le nom de l'empreinte de référence (ex: 101_1.tif): ")
#         if reference_image not in all_images:
#             print(f"L'empreinte {reference_image} n'existe pas dans le dossier.")
#         else:
#             compare_with_reference(reference_image, folder, all_images)
#     else:
#         print("Choix invalide.")