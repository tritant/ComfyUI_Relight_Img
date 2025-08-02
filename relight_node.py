import torch
import numpy as np
import cv2
import folder_paths 
import os
from PIL import Image

# Imports pour le serveur sécurisé
import http.server
import socketserver
import threading

# Le bloc manquant, remis à sa place
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Variable globale pour s'assurer que le serveur n'est démarré qu'une seule fois
preview_server_thread = None
PREVIEW_SERVER_PORT = 8189 # Port pour notre serveur

def start_preview_server():
    global preview_server_thread
    if preview_server_thread is None or not preview_server_thread.is_alive():
        class SecureHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=folder_paths.get_temp_directory(), **kwargs)
            def do_GET(self):
                if self.path == '/' or self.path.endswith('/'):
                    self.send_error(403, "Directory listing is not allowed")
                    return
                super().do_GET()
            def log_message(self, format, *args):
                return
        address = ("127.0.0.1", PREVIEW_SERVER_PORT)
        httpd = socketserver.TCPServer(address, SecureHandler)
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
        preview_server_thread = thread
        print(f"[Relight Node] Serveur d'aperçu sécurisé démarré sur http://127.0.0.1:{PREVIEW_SERVER_PORT}")


class RelightNode:
    OUTPUT_NODE = True 
    
    LIGHT_TYPES = ["Point", "Neon"]
    DEBUG_VIEWS = ["Final Image", "Lighting Map Only", "Mask Lighting Only", "Mask Virtual Normals", "Light Shape Viz"]

    @classmethod
    def INPUT_TYPES(s):
        if not SCIPY_AVAILABLE: 
            return {"required": {"error": ("STRING", {"default": "ERREUR: SciPy n'est pas installé. Exécutez 'pip install scipy' et redémarrez ComfyUI.", "multiline": True})}}
        
        return {
            "required": {
                "image": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "debug_view": (s.DEBUG_VIEWS, ),
                "preserve_color": ("BOOLEAN", {"default": True}),
                "ambient_light": ("FLOAT", {"default": -0.20, "min": -1.0, "max": 1.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 5.5, "min": 0.1, "max": 20.0, "step": 0.1}),
                "light_color": ("STRING", {"default": "#FFFFFF"}),
                "depth_scale": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "median_filter_size": ("INT", {"default": 31, "min": 1, "max": 31, "step": 2}),
                "dithering_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.1}),
                "spacer_1": ("STRING", {"multiline": False, "default": ""}),
                "mask_enabled": ("BOOLEAN", {"default": True}),
                "brush_intensity": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brush_softness": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "mask_intensity_mult": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mask_gamma_mult": ("FLOAT", {"default": 0.14, "min": 0.0, "max": 2.0, "step": 0.01}),
                "spacer_2": ("STRING", {"multiline": False, "default": ""}),
                "light_type_1": (s.LIGHT_TYPES, {"default": "Point"}),
                "light_x_1": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_y_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_z_1": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "intensity_1": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "point_size_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "neon_length_1": ("FLOAT", {"default": 0.77, "min": 0.0, "max": 2.0, "step": 0.01}),
                "neon_angle_1": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "ui_anchor": ("STRING", {"multiline": True, "default": ""}),
                "enable_light_2": ("BOOLEAN", {"default": False}),
                "light_type_2": (s.LIGHT_TYPES, {"default": "Point"}),
                "light_x_2": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_y_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "light_z_2": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "intensity_2": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "point_size_2": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "neon_length_2": ("FLOAT", {"default": 0.77, "min": 0.0, "max": 2.0, "step": 0.01}),
                "neon_angle_2": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            },
            "optional": { "mask": ("MASK",) }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "relight_image"
    CATEGORY = "Image/Lighting"

    @classmethod
    def IS_CHANGED(self, **kwargs): 
        return float("NaN")

    def parse_hex_color(self, hex_color: str) -> np.ndarray:
        hex_color = hex_color.lstrip('#');
        if len(hex_color) != 6: return np.array([1.0, 1.0, 1.0], dtype=np.float32)
        try:
            r = int(hex_color[0:2], 16) / 255.0; g = int(hex_color[2:4], 16) / 255.0; b = int(hex_color[4:6], 16) / 255.0
            return np.array([r, g, b], dtype=np.float32)
        except ValueError: return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray: return tensor.squeeze(0).cpu().numpy()

    def numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        if array.ndim == 2: array = np.expand_dims(array, axis=-1)
        if array.shape[2] == 1: array = np.repeat(array, 3, axis=-1)
        return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
        
    def tensor_to_pil(self, tensor):
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def relight_image(self, image: torch.Tensor, normal_map: torch.Tensor, depth_map: torch.Tensor,
                      debug_view: str, preserve_color: bool, ambient_light: float, gamma: float, light_color: str,
                      depth_scale: float, median_filter_size: int, dithering_strength: float, spacer_1: str, 
                      mask_enabled: bool, brush_intensity: float, brush_softness: float, mask_intensity_mult: float, mask_gamma_mult: float, spacer_2: str,
                      
                      light_type_1: str, light_x_1: float, light_y_1: float, light_z_1: float, intensity_1: float, point_size_1: float, neon_length_1: float, neon_angle_1: float,
                      ui_anchor: str,
                      enable_light_2: bool, light_type_2: str, light_x_2: float, light_y_2: float, light_z_2: float, intensity_2: float, point_size_2: float, neon_length_2: float, neon_angle_2: float,
                      mask: torch.Tensor = None):
        
        start_preview_server()
        preview_image_pil = self.tensor_to_pil(image)
        preview_filename = "relightpreview.png"
        preview_filepath = os.path.join(folder_paths.get_temp_directory(), preview_filename)
        preview_image_pil.save(preview_filepath)
        full_url = f"http://127.0.0.1:{PREVIEW_SERVER_PORT}/{preview_filename}"

        # VOTRE LOGIQUE DE TRAITEMENT ORIGINALE ET COMPLÈTE
        img_np = self.tensor_to_numpy(image).astype(np.float32); height, width, _ = img_np.shape
        normal_np = self.tensor_to_numpy(normal_map).astype(np.float32)
        depth_np = self.tensor_to_numpy(depth_map).astype(np.float32)
        np.nan_to_num(normal_np, copy=False, nan=0.0); np.nan_to_num(depth_np, copy=False, nan=0.0)
        if normal_np.shape[:2] != (height, width): normal_np = cv2.resize(normal_np, (width, height), interpolation=cv2.INTER_LINEAR)
        if len(depth_np.shape) < 3 or depth_np.shape[:2] != (height, width): depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_LINEAR)
        if len(depth_np.shape) == 2: depth_np = np.expand_dims(depth_np, axis=-1)
        normal_vectors = (normal_np * 2.0) - 1.0
        xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        pixel_z = (1.0 - depth_np[..., 0]) * depth_scale
        pixel_positions = np.stack((xx.astype(np.float32), yy.astype(np.float32), pixel_z.astype(np.float32)), axis=-1)
        light_vectors_1 = np.zeros_like(pixel_positions)
        if light_type_1 == "Point":
            light_pos_1 = np.array([light_x_1 * width, light_y_1 * height, light_z_1 * max(width, height)])
            light_vectors_1 = light_pos_1 - pixel_positions
        elif light_type_1 == "Neon":
            center_x_1, center_y_1 = light_x_1 * width, light_y_1 * height; light_pos_z_1 = light_z_1 * max(width, height); angle_rad_1 = np.deg2rad(neon_angle_1); half_len_1 = (neon_length_1 / 2.0) * max(width, height); dx_1, dy_1 = half_len_1 * np.cos(angle_rad_1), half_len_1 * np.sin(angle_rad_1); p1_1 = np.array([center_x_1 - dx_1, center_y_1 - dy_1, light_pos_z_1]); p2_1 = np.array([center_x_1 + dx_1, center_y_1 + dy_1, light_pos_z_1]); line_vec_1 = p2_1 - p1_1; line_len_sq_1 = np.sum(line_vec_1**2)
            if line_len_sq_1 == 0: line_len_sq_1 = 1.0
            pixel_to_p1_1 = pixel_positions - p1_1; t_1 = np.sum(pixel_to_p1_1 * line_vec_1, axis=2) / line_len_sq_1; t_1 = np.clip(t_1, 0.0, 1.0); closest_points_1 = p1_1 + np.expand_dims(t_1, axis=2) * line_vec_1; light_vectors_1 = closest_points_1 - pixel_positions
        dist_1 = np.linalg.norm(light_vectors_1, axis=2, keepdims=True); dist_1[dist_1 == 0] = 1.0
        normalized_light_vectors_1 = light_vectors_1 / dist_1
        normalized_light_vectors_2 = np.zeros_like(pixel_positions)
        if enable_light_2:
            light_vectors_2 = np.zeros_like(pixel_positions)
            if light_type_2 == "Point":
                light_pos_2 = np.array([light_x_2 * width, light_y_2 * height, light_z_2 * max(width, height)]); light_vectors_2 = light_pos_2 - pixel_positions
            elif light_type_2 == "Neon":
                center_x_2, center_y_2 = light_x_2 * width, light_y_2 * height; light_pos_z_2 = light_z_2 * max(width, height); angle_rad_2 = np.deg2rad(neon_angle_2); half_len_2 = (neon_length_2 / 2.0) * max(width, height); dx_2, dy_2 = half_len_2 * np.cos(angle_rad_2), half_len_2 * np.sin(angle_rad_2); p1_2 = np.array([center_x_2 - dx_2, center_y_2 - dy_2, light_pos_z_2]); p2_2 = np.array([center_x_2 + dx_2, center_y_2 + dy_2, light_pos_z_2]); line_vec_2 = p2_2 - p1_2; line_len_sq_2 = np.sum(line_vec_2**2)
                if line_len_sq_2 == 0: line_len_sq_2 = 1.0
                pixel_to_p1_2 = pixel_positions - p1_2; t_2 = np.sum(pixel_to_p1_2 * line_vec_2, axis=2) / line_len_sq_2; t_2 = np.clip(t_2, 0.0, 1.0); closest_points_2 = p1_2 + np.expand_dims(t_2, axis=2) * line_vec_2; light_vectors_2 = closest_points_2 - pixel_positions
            dist_2 = np.linalg.norm(light_vectors_2, axis=2, keepdims=True); dist_2[dist_2 == 0] = 1.0
            normalized_light_vectors_2 = light_vectors_2 / dist_2
        dot_product_1 = np.sum(normal_vectors * normalized_light_vectors_1, axis=2)
        if light_type_1 == "Point" and point_size_1 > 0: diffuse_term_1 = (dot_product_1 + point_size_1) / (1 + point_size_1)
        else: diffuse_term_1 = dot_product_1
        raw_light_1 = np.maximum(0, diffuse_term_1) * intensity_1
        raw_light_2 = np.zeros_like(raw_light_1)
        if enable_light_2:
            dot_product_2 = np.sum(normal_vectors * normalized_light_vectors_2, axis=2)
            if light_type_2 == "Point" and point_size_2 > 0: diffuse_term_2 = (dot_product_2 + point_size_2) / (1 + point_size_2)
            else: diffuse_term_2 = dot_product_2
            raw_light_2 = np.maximum(0, diffuse_term_2) * intensity_2
        base_raw_light = raw_light_1 + raw_light_2
        np.clip(base_raw_light, 0.0, 150.0, out=base_raw_light)
        base_compressed_light = np.zeros_like(base_raw_light, dtype=np.float32)
        if np.any(base_raw_light > 0): base_compressed_light[base_raw_light > 0] = np.power(base_raw_light[base_raw_light > 0], 1.0 / gamma)
        final_lighting_map = base_compressed_light
        shape_normals_vis = np.zeros((height, width, 3), dtype=np.float32); lighting_map_shape_vis = np.zeros((height, width), dtype=np.float32)
        if mask is not None and mask_enabled:
            mask_np = self.tensor_to_numpy(mask).squeeze();
            if mask_np.shape[:2] != (height, width):
               mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)            
            mask_np_binary = (mask_np > 0.5).astype(np.uint8)
            shape_normals = np.zeros((height, width, 3), dtype=np.float32)
            if np.any(mask_np_binary > 0):
                if mask_np_binary.shape[:2] != (height, width): mask_np_binary = cv2.resize(mask_np_binary, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_np_binary > 0; dist_transform = distance_transform_edt(mask_np_binary); max_dist = np.max(dist_transform); normalized_dist = dist_transform / max_dist if max_dist > 0 else 0; grad_y, grad_x = np.gradient(normalized_dist)
                shape_normals[mask_bool, 0] = -grad_x[mask_bool]; shape_normals[mask_bool, 1] = -grad_y[mask_bool]; shape_normals[mask_bool, 2] = 1.0
                norm_sn = np.linalg.norm(shape_normals, axis=2, keepdims=True); norm_sn[norm_sn == 0] = 1; shape_normals /= norm_sn
            shape_normals_vis = (shape_normals + 1.0) / 2.0
            mask_dot_product_1 = np.sum(shape_normals * normalized_light_vectors_1, axis=2)
            if light_type_1 == "Point" and point_size_1 > 0: mask_diffuse_1 = (mask_dot_product_1 + point_size_1) / (1 + point_size_1)
            else: mask_diffuse_1 = mask_dot_product_1
            mask_raw_light_1 = np.maximum(0, mask_diffuse_1) * intensity_1 * mask_intensity_mult
            mask_raw_light_2 = np.zeros_like(mask_raw_light_1)
            if enable_light_2:
                mask_dot_product_2 = np.sum(shape_normals * normalized_light_vectors_2, axis=2)
                if light_type_2 == "Point" and point_size_2 > 0: mask_diffuse_2 = (mask_dot_product_2 + point_size_2) / (1 + point_size_2)
                else: mask_diffuse_2 = mask_dot_product_2
                mask_raw_light_2 = np.maximum(0, mask_diffuse_2) * intensity_2 * mask_intensity_mult
            mask_total_raw_light = mask_raw_light_1 + mask_raw_light_2
            np.clip(mask_total_raw_light, 0.0, 150.0, out=mask_total_raw_light)
            mask_compressed_light = np.zeros_like(mask_total_raw_light, dtype=np.float32)
            final_mask_gamma = gamma * mask_gamma_mult;
            if final_mask_gamma < 0.1: final_mask_gamma = 0.1
            if np.any(mask_total_raw_light > 0): mask_compressed_light[mask_total_raw_light > 0] = np.power(mask_total_raw_light[mask_total_raw_light > 0], 1.0 / final_mask_gamma)
            lighting_map_shape_vis = mask_compressed_light
            blur_sigma = brush_softness
            if blur_sigma > 0:
                k_size = int(blur_sigma * 4) + 1; k_size = k_size if k_size % 2 != 0 else k_size + 1; blend_mask = cv2.GaussianBlur(mask_np.astype(np.float32), (k_size, k_size), blur_sigma)
            else: blend_mask = mask_np.astype(np.float32)
            blend_mask *= brush_intensity
            final_lighting_map = (base_compressed_light * (1.0 - blend_mask)) + (mask_compressed_light * blend_mask)
        final_lighting_map_expanded = np.expand_dims(final_lighting_map, axis=-1)
        if median_filter_size > 1:
            final_lighting_map_expanded = np.nan_to_num(final_lighting_map_expanded, nan=0.0)
            k_size = median_filter_size if median_filter_size % 2 != 0 else median_filter_size + 1; squeezed_map = np.squeeze(final_lighting_map_expanded); min_val, max_val = np.min(squeezed_map), np.max(squeezed_map)
            if max_val > min_val:
                normalized_map = (squeezed_map - min_val) / (max_val - min_val); map_8bit = (normalized_map * 255).astype(np.uint8); filtered_8bit = cv2.medianBlur(map_8bit, k_size); filtered_normalized = filtered_8bit.astype(np.float32) / 255.0; restored_map = filtered_normalized * (max_val - min_val) + min_val; final_lighting_map_expanded = np.expand_dims(restored_map, axis=-1)
        final_map_with_ambient = np.squeeze(final_lighting_map_expanded) + ambient_light
        light_color_rgb = self.parse_hex_color(light_color); is_white_light = np.all(light_color_rgb == 1.0)
        output_np = np.zeros_like(img_np, dtype=np.float32)
        if debug_view == "Final Image":
            colored_lighting_map = np.expand_dims(final_map_with_ambient, axis=-1) * light_color_rgb
            if preserve_color and is_white_light:
                hls_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2HLS); hls_image[:, :, 1] = np.clip(hls_image[:, :, 1] * final_map_with_ambient, 0.0, 1.0); output_np = cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)
            else: output_np = np.clip(img_np * colored_lighting_map, 0.0, 1.0)
        elif debug_view == "Lighting Map Only": output_np = final_map_with_ambient
        elif debug_view == "Mask Lighting Only": output_np = lighting_map_shape_vis
        elif debug_view == "Mask Virtual Normals": output_np = shape_normals_vis
        if dithering_strength > 0 and debug_view == "Final Image":
            noise = np.random.normal(0.0, dithering_strength / 255.0, output_np.shape).astype(np.float32)
            output_np += noise
        output_np = np.nan_to_num(np.clip(output_np, 0.0, 1.0))
        
        return {
            "result": (self.numpy_to_tensor(output_np),),
            "ui": { "previews": [full_url] }
        }

NODE_CLASS_MAPPINGS = { "RelightNode": RelightNode }
NODE_DISPLAY_NAME_MAPPINGS = { "RelightNode": "Advanced Relight Node" }