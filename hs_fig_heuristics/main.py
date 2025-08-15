from PIL import Image
from sklearn.model_selection import train_test_split
from mobilenet import create_model
from sklearn.metrics import classification_report
from genetic_algorithm import (
    crear_poblacion_inicial,
    crear_individuo,
    seleccion_por_torneo,
    cruce,
    mutacion,
    NUM_GENES,
    POPULATION_SIZE,
    NUM_GENERATIONS,
    TOURNAMENT_SIZE,
    MUTATION_RATE,
    ELITISM_COUNT,
    GENE_POOL
)

import pprint
import cv2
import numpy as np
import radiometric_corrections
import json
import threading


wavelengths = [397.01, 398.32, 399.63, 400.93, 402.24, 403.55, 404.86, 406.17, 407.48, 408.79, 410.10, 411.41, 412.72, 414.03, 415.34, 416.65, 417.96, 419.27, 420.58, 421.90, 423.21, 424.52, 425.83,
               427.15, 428.46, 429.77, 431.09, 432.40, 433.71, 435.03, 436.34, 437.66, 438.97, 440.29, 441.60, 442.92, 444.23, 445.55, 446.87, 448.18, 449.50, 450.82, 452.13, 453.45, 454.77, 456.09, 457.40,
               458.72, 460.04, 461.36, 462.68, 464.00, 465.32, 466.64, 467.96, 469.28, 470.60, 471.92, 473.24, 474.56, 475.88, 477.20, 478.52, 479.85, 481.17, 482.49, 483.81, 485.14, 486.46, 487.78, 489.11,
               490.43, 491.75, 493.08, 494.40, 495.73, 497.05, 498.38, 499.70, 501.03, 502.35, 503.68, 505.01, 506.33, 507.66, 508.99, 510.31, 511.64, 512.97, 514.30, 515.63, 516.95, 518.28, 519.61, 520.94,
               522.27, 523.60, 524.93, 526.26, 527.59, 528.92, 530.25, 531.58, 532.91, 534.25, 535.58, 536.91, 538.24, 539.57, 540.91, 542.24, 543.57, 544.90, 546.24, 547.57, 548.91, 550.24, 551.57, 552.91,
               554.24, 555.58, 556.91, 558.25, 559.59, 560.92, 562.26, 563.59, 564.93, 566.27, 567.61, 568.94, 570.28, 571.62, 572.96, 574.30, 575.63, 576.97, 578.31, 579.65, 580.99, 582.33, 583.67, 585.01,
               586.35, 587.69, 589.03, 590.37, 591.71, 593.06, 594.40, 595.74, 597.08, 598.42, 599.77, 601.11, 602.45, 603.80, 605.14, 606.48, 607.83, 609.17, 610.52, 611.86, 613.21, 614.55, 615.90, 617.24,
               618.59, 619.94, 621.28, 622.63, 623.98, 625.32, 626.67, 628.02, 629.37, 630.71, 632.06, 633.41, 634.76, 636.11, 637.46, 638.81, 640.16, 641.51, 642.86, 644.21, 645.56, 646.91, 648.26, 649.61,
               650.96, 652.31, 653.67, 655.02, 656.37, 657.72, 659.08, 660.43, 661.78, 663.14, 664.49, 665.84, 667.20, 668.55, 669.91, 671.26, 672.62, 673.97, 675.33, 676.68, 678.04, 679.40, 680.75, 682.11,
               683.47, 684.82, 686.18, 687.54, 688.90, 690.25, 691.61, 692.97, 694.33, 695.69, 697.05, 698.41, 699.77, 701.13, 702.49, 703.85, 705.21, 706.57, 707.93, 709.29, 710.65, 712.02, 713.38, 714.74, 
               716.10, 717.47, 718.83, 720.19, 721.56, 722.92, 724.28, 725.65, 727.01, 728.38, 729.74, 731.11, 732.47, 733.84, 735.20, 736.57, 737.93, 739.30, 740.67, 742.03, 743.40, 744.77, 746.14, 747.50,
               748.87, 750.24, 751.61, 752.98, 754.35, 755.72, 757.09, 758.46, 759.83, 761.20, 762.57, 763.94, 765.31, 766.68, 768.05, 769.42, 770.79, 772.17, 773.54, 774.91, 776.28, 777.66, 779.03, 780.40,
               781.78, 783.15, 784.52, 785.90, 787.27, 788.65, 790.02, 791.40, 792.77, 794.15, 795.52, 796.90, 798.28, 799.65, 801.03, 802.41, 803.78, 805.16, 806.54, 807.92, 809.30, 810.67, 812.05, 813.43,
               814.81, 816.19, 817.57, 818.95, 820.33, 821.71, 823.09, 824.47, 825.85, 827.23, 828.61, 830.00, 831.38, 832.76, 834.14, 835.53, 836.91, 838.29, 839.67, 841.06, 842.44, 843.83, 845.21, 846.59,
               847.98, 849.36, 850.75, 852.13, 853.52, 854.91, 856.29, 857.68, 859.06, 860.45, 861.84, 863.23, 864.61, 866.00, 867.39, 868.78, 870.16, 871.55, 872.94, 874.33, 875.72, 877.11, 878.50, 879.89,
               881.28, 882.67, 884.06, 885.45, 886.84, 888.23, 889.63, 891.02, 892.41, 893.80, 895.19, 896.59, 897.98, 899.37, 900.77, 902.16, 903.55, 904.95, 906.34, 907.74, 909.13, 910.53, 911.92, 913.32,
               914.71, 916.11, 917.50, 918.90, 920.30, 921.69, 923.09, 924.49, 925.89, 927.28, 928.68, 930.08, 931.48, 932.88, 934.28, 935.68, 937.08, 938.48, 939.88, 941.28, 942.68, 944.08, 945.48, 946.88,
               948.28, 949.68, 951.08, 952.48, 953.89, 955.29, 956.69, 958.09, 959.50, 960.90, 962.30, 963.71, 965.11, 966.52, 967.92, 969.33, 970.73, 972.14, 973.54, 974.95, 976.35, 977.76, 979.16, 980.57,
               981.98, 983.38, 984.79, 986.20, 987.61, 989.02, 990.42, 991.83, 993.24, 994.65, 996.06, 997.47, 998.88, 1000.29, 1001.70, 1003.11, 1004.52]
IMG_SIZE = (224, 224)


_masks_cache = {}
_masks_cache_lock = threading.Lock()

def masks_per_figure(mask_image_path: str) -> list[np.ndarray]:
    """
    Gets the individual masks per figure in the image

    Note: Expects an image of color masks with black background

    Returns:
        list: A list of masks
    """
    with _masks_cache_lock:
        if mask_image_path in _masks_cache:
            # Return a copy to avoid accidental modification
            return [mask.copy() for mask in _masks_cache[mask_image_path]]

    masks = []
    image = cv2.imread(mask_image_path)

    if image is None:
        raise Exception(f'Not image found in {mask_image_path}')
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    print(f"Se encontraron {len(unique_colors) - 1} frutas (colores únicos).")
    for color in unique_colors:
        if np.all(color == [0, 0, 0]):
            continue
        
        color_maks = cv2.inRange(image, color, color)
        contours, _ = cv2.findContours(color_maks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            individual_mask = np.zeros_like(grey_img)
            cv2.drawContours(individual_mask, [contour], -1, (255), thickness=cv2.FILLED)
            masks.append(individual_mask)

    with _masks_cache_lock:
        _masks_cache[mask_image_path] = [mask.copy() for mask in masks]

    return masks


def build_fake_rgb_matrix(cube: np.ndarray, bands: list[float]) -> np.ndarray:
    """
    Builds a fake RGB matrix from a hyperspectral cube and specified bands.

    Args:
        cube (np.ndarray): Hyperspectral cube with shape (rows, columns, bands).
        bands (list[float]): List of band indices to use for the RGB channels.

    Returns:
        np.ndarray: Fake RGB image with shape (rows, columns, 3).
    """
    if len(bands) != 3:
        raise ValueError("Exactly three bands must be provided for RGB channels.")
    
    rgb_image = np.zeros((cube.shape[0], cube.shape[1], 3), dtype=np.float64)
    rgb_image[:, :, 0] = cube[:, :, wavelengths.index(bands[0])]  # Red channel
    rgb_image[:, :, 1] = cube[:, :, wavelengths.index(bands[1])]  # Green channel
    rgb_image[:, :, 2] = cube[:, :, wavelengths.index(bands[2])]  # Blue channel

    return rgb_image


def build_fake_rgb_image(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Builds a fake RGB image from a hyperspectral matrix and a mask.

    Args:
        matrix (np.ndarray): Hyperspectral matrix with shape (rows, columns, bands).
        mask (np.ndarray): Mask with shape (rows, columns).

    Returns:
        np.ndarray: Fake RGB image with shape (rows, columns, 3).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0]) # type: ignore
    cropped_matrix = matrix[y:y+h, x:x+w, :]
    # 8bit conversion
    cropped_matrix = (cropped_matrix * 255).astype(np.uint8)
    # Create a mask for the cropped area
    cropped_mask = mask[y:y+h, x:x+w]
    final_cut = cv2.bitwise_and(cropped_matrix, cropped_matrix, mask=cropped_mask)
    img = Image.fromarray(final_cut, 'RGB').resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return np.array(img)


def process_images(images_configs: list, bands: list[float]) -> list[np.ndarray]:
    """
    Processes images based on the provided configuration and bands.

    Args:
        images_config (dict): Configuration dictionary containing image paths.
        bands (list[float]): List of band indices to use for RGB channels.

    Returns:
        list[np.ndarray]: List of processed RGB images.
    """
    fake_images = []

    for img_conf in images_configs:
        masks = masks_per_figure(img_conf['mask'])
        cube = radiometric_corrections.black_white_correction(
            img_conf['hs_image_hdr'], 
            img_conf['black_ref_hdr'], 
            img_conf['white_ref_hdr']
        )
        fake_matrix = build_fake_rgb_matrix(cube, bands)
        for mask in masks:
            fake_rgb_image = build_fake_rgb_image(fake_matrix, mask)
            fake_images.append(fake_rgb_image)

    return fake_images


def objective_function(individual: list[float]) -> float:
    # individual = [701.13, 702.49, 703.85]
    with open('/Users/israel/Projects/hs_fig_heuristics/image_locations.json', 'r') as file:
        images_config = json.load(file)
    
    healty_figs = process_images(images_config['0'], individual)
    infected_figs = process_images(images_config['1'], individual)
    X = np.array(healty_figs + infected_figs)
    Y = np.array([0] * len(healty_figs) + [1] * len(infected_figs))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)
    
    model, history = create_model(2, X_train, Y_train, X_val, Y_val)

    prediction_probabilities = model.predict(X_test)
    predicted_classes = np.argmax(prediction_probabilities, axis=1)
    report = classification_report(Y_test, predicted_classes, target_names=['Healthy', 'Infected'], output_dict=True)
    score = report['macro avg']['f1-score']
    print(f"F1 Score: {score:.4f} | Individual: {individual}")
    return float(score)


def main():
    # 1. Creamos la población inicial
    poblacion = crear_poblacion_inicial()
    mejor_individuo_global = None
    mejor_fitness_global = -1
    best_individuals_fitness_by_generation = []

    print("--- Iniciando Algoritmo Genético ---")

    # 2. Ciclo de generaciones
    for i in range(NUM_GENERATIONS):
        # 3. Evaluamos la aptitud de cada individuo
        poblacion_con_fitness = [(ind, objective_function(ind)) for ind in poblacion]
        
        # Ordenamos la población por fitness (de mayor a menor)
        poblacion_con_fitness.sort(key=lambda item: item[1], reverse=True)
        
        # Actualizamos el mejor individuo encontrado hasta ahora
        mejor_de_generacion = poblacion_con_fitness[0]
        if mejor_de_generacion[1] > mejor_fitness_global:
            mejor_individuo_global = mejor_de_generacion[0]
            mejor_fitness_global = mejor_de_generacion[1]
            
        print(f"Generación {i+1}/{NUM_GENERATIONS} | Mejor Fitness: {mejor_de_generacion[1]:.2f} | Mejor Individuo: {mejor_de_generacion[0]}")
        best_individuals_fitness_by_generation.append((mejor_de_generacion[1], mejor_de_generacion[0]))

        # 4. Creamos la siguiente generación
        siguiente_generacion = []
        
        # 4.1. Elitismo: Los mejores pasan directamente
        for j in range(ELITISM_COUNT):
            siguiente_generacion.append(poblacion_con_fitness[j][0])
            
        # 4.2. Creamos el resto de la nueva generación
        while len(siguiente_generacion) < POPULATION_SIZE:
            # Selección de padres
            padre1 = seleccion_por_torneo(poblacion_con_fitness)
            padre2 = seleccion_por_torneo(poblacion_con_fitness)
            
            # Cruce
            hijo1, hijo2 = cruce(padre1, padre2)
            
            # Mutación
            hijo1 = mutacion(hijo1)
            hijo2 = mutacion(hijo2)
            
            siguiente_generacion.append(hijo1)
            if len(siguiente_generacion) < POPULATION_SIZE:
                siguiente_generacion.append(hijo2)
                
        # 5. Reemplazamos la población antigua por la nueva
        poblacion = siguiente_generacion

    print("\n--- Algoritmo Finalizado ---")
    print(f"Mejor individuo encontrado: {mejor_individuo_global}")
    print(f"Valor de fitness (F1 Score): {mejor_fitness_global:.4f}")
    
    print("Mejores individuos por generación:")
    pprint.pprint(best_individuals_fitness_by_generation)
    
    return mejor_individuo_global, mejor_fitness_global


if __name__ == '__main__':
    best_individual, best_fitness = main()
