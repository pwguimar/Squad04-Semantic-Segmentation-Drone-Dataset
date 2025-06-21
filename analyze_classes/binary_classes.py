import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def _validate_masks_directory(directory_path):   
    print(f"\n--- Verificando o Diretório ---")
    if not os.path.isdir(directory_path):
        print(f"ERRO: O diretório '{directory_path}' NÃO FOI ENCONTRADO.")
        return [], 0

    mask_files_list = [f for f in os.listdir(directory_path) if f.lower().endswith((".png", ".jpg"))]
    total_found_masks = len(mask_files_list)

    if total_found_masks == 0:
        print(f"ERRO: NENHUM arquivo de imagem encontrado em '{directory_path}'.")
        return [], 0

    print(f"SUCESSO: Encontrados {total_found_masks} imagens.")
    return mask_files_list, total_found_masks

def _count_pixels_in_masks(masks_directory, mask_files_list, rgb_to_binary_id_map, binary_id_to_name_map, color_tolerance=5): 
    class_pixel_counts_dict = {name: 0 for name in binary_id_to_name_map.values()}
    
    print(f"\nIniciando a contagem de pixels nas máscaras...")
    processed_count = 0

    for mask_filename in mask_files_list:
        mask_path = os.path.join(masks_directory, mask_filename)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_COLOR) 
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

        if mask_image is None:
            print(f"Aviso: Não foi possível carregar a máscara '{mask_filename}'. Retornou None. Formato inválido ou corrompido?")
            continue
        
        if processed_count == 0:            
            unique_rgb_tuples_debug = []
            reshaped_image_debug = mask_image.reshape(-1, 3)
            unique_rgb_set_debug = {tuple(pixel) for pixel in reshaped_image_debug}
            unique_rgb_tuples_debug = sorted(list(unique_rgb_set_debug))

            print(f"Todos os valores RGB únicos nesta máscara:")
            for rgb_val in unique_rgb_tuples_debug:
                print(f"  {rgb_val}")
            print("-" * 50)
       
        remapped_mask = np.full(mask_image.shape[:2], 255, dtype=np.uint8)

        for rgb_tuple, binary_id in rgb_to_binary_id_map.items():
            target_r, target_g, target_b = rgb_tuple
            
            min_r, max_r = max(0, target_r - color_tolerance), min(255, target_r + color_tolerance)
            min_g, max_g = max(0, target_g - color_tolerance), min(255, target_g + color_tolerance)
            min_b, max_b = max(0, target_b - color_tolerance), min(255, target_b + color_tolerance)
            
            match_r = (mask_image[:,:,0] >= min_r) & (mask_image[:,:,0] <= max_r)
            match_g = (mask_image[:,:,1] >= min_g) & (mask_image[:,:,1] <= max_g)
            match_b = (mask_image[:,:,2] >= min_b) & (mask_image[:,:,2] <= max_b)
            
            matches_color_with_tolerance = match_r & match_g & match_b
            
            remapped_mask[matches_color_with_tolerance] = binary_id
        
        if processed_count == 0:
            unique_vals_remapped = np.unique(remapped_mask)
            print(f"Valores de pixel únicos APÓS remapeamento (máx 20): {unique_vals_remapped[:20]}")
            if not np.any(remapped_mask == 0) and not np.any(remapped_mask == 1):
                print("Aviso: Após o remapeamento, a máscara não contém 0s ou 1s esperados.")
            if 255 in unique_vals_remapped:
                 print(f"Aviso: Ainda existem {np.sum(remapped_mask == 255)} pixels não mapeados (valor 255).")
            print("-" * 50)

        unique_target_ids, counts = np.unique(remapped_mask, return_counts=True)
        
        for i, target_id_val in enumerate(unique_target_ids):
            if target_id_val != 255 and target_id_val in binary_id_to_name_map:
                class_name = binary_id_to_name_map[target_id_val]
                class_pixel_counts_dict[class_name] += counts[i]
            elif target_id_val == 255:
                pass 
            else:
                print(f"Aviso: Máscara '{mask_filename}' contém pixels com ID de classe binária '{target_id_val}' desconhecido (não 0, 1 ou 255).")

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processados {processed_count}/{len(mask_files_list)} arquivos...")

    print("\n--- Processamento Concluído ---")
    return class_pixel_counts_dict

def _calculate_and_print_distribution(pixel_counts, min_threshold_percentage_val, title_prefix=""):
    total_pixels_counted = sum(pixel_counts.values())

    if total_pixels_counted == 0:
        print(f"\nERRO FINAL: Nenhum pixel foi contado após o processamento para {title_prefix}.")
        return [], {}
    else:
        percentages_dict = {
            class_name: (count / total_pixels_counted) * 100
            for class_name, count in pixel_counts.items()
        }

        print(f"\n--- Distribuição de Classes ({title_prefix}) ---")
        sorted_info = sorted(pixel_counts.items(), key=lambda item: item[1], reverse=True)

        for class_name, count in sorted_info:
            print(f"Classe: {class_name:<15} | Pixels: {count:<15} | Porcentagem: {percentages_dict[class_name]:.2f}%")

        print(f"\n--- Possíveis Classes Minoritárias ({title_prefix} - abaixo de {min_threshold_percentage_val:.1f}%) ---")
        minority_found = False
        for class_name, percentage in sorted(percentages_dict.items(), key=lambda item: item[1]):
            if percentage < min_threshold_percentage_val:
                print(f"Classe: {class_name:<15} | Porcentagem: {percentage:.2f}%")
                minority_found = True
        if not minority_found:
            print(f"Nenhuma classe abaixo do threshold de {min_threshold_percentage_val:.1f}% foi identificada para {title_prefix}.")
        
        return sorted_info, percentages_dict

def _plot_distribution(sorted_class_data, percentages_data, plot_title="Distribuição de Pixels por Classe"):
    class_names_plot = [item[0] for item in sorted_class_data]
    percentages_plot = [percentages_data[name] for name in class_names_plot]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names_plot, percentages_plot, color=['lightcoral', 'lightgreen'])
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Porcentagem de Pixels (%)", fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

binary_classes = {
    0: {0, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    1: {1, 2, 3, 4, 9}
}

rgb_to_binary_id = {
    (204, 153, 255): 0, # 'obstacles'
    (0, 153, 153): 1   # CORRIGIDO: Esta é a cor real para 'landing-zones' / ID 1
}

binary_id_to_name = {
    0: 'obstacles',
    1: 'landing-zones'
}


# 2 - DEFINIÇÃO DO DIRETÓRIO DE MÁSCARAS BINÁRIAS
binary_masks_dir = "" # Defina o caminho para o diretório de máscaras binárias aqui

if __name__ == "__main__":
    binary_original_id_to_target_id = None 
    
    binary_mask_files, total_binary_masks_found = _validate_masks_directory(binary_masks_dir)
    
    if total_binary_masks_found > 0:
        # Passa o novo mapeamento rgb_to_binary_id e a tolerância de cor (pode ser menor agora, talvez 0 ou 1)
        binary_pixel_counts_result = _count_pixels_in_masks(binary_masks_dir, binary_mask_files, rgb_to_binary_id, binary_id_to_name, color_tolerance=1) # Tolerância ajustada para 1
        
        min_threshold_percentage = 2.0
        sorted_binary_class_info_final, binary_class_percentages_final = _calculate_and_print_distribution(binary_pixel_counts_result, min_threshold_percentage, "em Pixels Binários")
        
        if sorted_binary_class_info_final:
            _plot_distribution(sorted_binary_class_info_final, binary_class_percentages_final, "Distribuição de Pixels por Classe Binária")