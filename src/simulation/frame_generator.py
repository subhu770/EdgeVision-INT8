"""Deterministic synthetic industrial defect generator.

Generates realistic clean metal and PCB surfaces as well as industrial defects
(cracks, scratches, and pits) with ground-truth bounding box coordinates.
Zero GPU or external dataset downloads required.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


DEFECT_CLASSES = ["crack", "scratch", "pit"]
CLASS_TO_ID = {name: i for i, name in enumerate(DEFECT_CLASSES)}
ID_TO_CLASS = {i: name for i, name in enumerate(DEFECT_CLASSES)}


class IndustrialDefectGenerator:
    """Generates synthetic industrial surface images with ground-truth defect annotations."""

    def __init__(self, width: int = 256, height: int = 256, seed: Optional[int] = 42):
        """Initialize the synthetic frame generator.

        Args:
            width: Output frame width in pixels.
            height: Output frame height in pixels.
            seed: Deterministic random seed for reproducibility.
        """
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int) -> None:
        """Reset the internal random state seed."""
        self.rng = np.random.RandomState(seed)

    def _generate_metal_surface(self) -> np.ndarray:
        """Generate brushed sheet metal surface with realistic grain and specular gradient."""
        # Base luminance for cold-rolled / brushed industrial steel
        base_val = self.rng.randint(140, 175)
        base = np.full((self.height, self.width), base_val, dtype=np.float32)

        # Directional horizontal brushed streaks
        streak_profile = self.rng.normal(0, 12, (self.height, 1)).astype(np.float32)
        streaks = np.tile(streak_profile, (1, self.width))
        
        # Specular lighting gradient (subtle illumination non-uniformity across the frame)
        x_grad = np.linspace(-15.0, 15.0, self.width, dtype=np.float32)
        y_grad = np.linspace(-10.0, 10.0, self.height, dtype=np.float32)
        grad_x, grad_y = np.meshgrid(x_grad, y_grad)
        
        # Fine grain microstructure noise
        grain_noise = self.rng.normal(0, 4.0, (self.height, self.width)).astype(np.float32)
        
        metal = base + streaks + grad_x + grad_y + grain_noise
        metal = np.clip(metal, 0, 255).astype(np.uint8)
        
        # Convert to 3-channel BGR image
        return cv2.cvtColor(metal, cv2.COLOR_GRAY2BGR)

    def _generate_pcb_surface(self) -> np.ndarray:
        """Generate PCB / solder-mask surface with traces and circular solder vias."""
        # Dark green solder mask background
        bg_color = np.array([35, 75, 40], dtype=np.uint8)  # BGR
        pcb = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)

        # Micro-texture noise
        noise = self.rng.normal(0, 3.0, (self.height, self.width, 3)).astype(np.float32)
        pcb = np.clip(pcb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Draw circuit trace lines (copper / gold paths)
        trace_color = (65, 145, 185)  # BGR gold/copper sheen
        num_traces = self.rng.randint(4, 8)
        for _ in range(num_traces):
            y_start = self.rng.randint(20, self.height - 20)
            x_mid = self.rng.randint(40, self.width - 40)
            y_mid = y_start + self.rng.choice([-20, 0, 20])
            y_mid = int(np.clip(y_mid, 10, self.height - 10))

            pts = np.array([
                [0, y_start],
                [x_mid, y_start],
                [x_mid + 20, y_mid],
                [self.width, y_mid]
            ], dtype=np.int32)
            cv2.polylines(pcb, [pts], isClosed=False, color=trace_color, thickness=2, lineType=cv2.LINE_AA)

        # Draw circular vias / solder pads
        num_vias = self.rng.randint(5, 10)
        via_rim_color = (80, 165, 210)
        via_hole_color = (20, 20, 20)
        for _ in range(num_vias):
            cx = self.rng.randint(25, self.width - 25)
            cy = self.rng.randint(25, self.height - 25)
            cv2.circle(pcb, (cx, cy), 5, via_rim_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(pcb, (cx, cy), 2, via_hole_color, -1, lineType=cv2.LINE_AA)

        return pcb

    def _draw_crack(self, img: np.ndarray) -> Optional[List[float]]:
        """Draw realistic non-linear jagged crack via stochastic 2D random walk.

        Returns bounding box [x1, y1, x2, y2] in pixels or None if invalid.
        """
        margin = 35
        cur_x = float(self.rng.randint(margin, self.width - margin))
        cur_y = float(self.rng.randint(margin, self.height - margin))
        
        # Main propagation direction angle
        angle = self.rng.uniform(0, 2 * np.pi)
        length_steps = self.rng.randint(20, 45)
        
        coords: List[Tuple[int, int]] = [(int(cur_x), int(cur_y))]
        crack_color = (25, 25, 25)
        halo_color = (190, 190, 190)

        for _ in range(length_steps):
            # Angular jitter mimicking brittle fracture mechanics
            angle += self.rng.uniform(-0.55, 0.55)
            step_size = self.rng.uniform(2.5, 4.5)
            cur_x += step_size * np.cos(angle)
            cur_y += step_size * np.sin(angle)
            
            # Clip bounds
            cur_x = np.clip(cur_x, 5.0, self.width - 5.0)
            cur_y = np.clip(cur_y, 5.0, self.height - 5.0)
            coords.append((int(cur_x), int(cur_y)))

        # Render crack with subtle specular halo highlight
        for i in range(len(coords) - 1):
            pt1 = coords[i]
            pt2 = coords[i + 1]
            # Draw slight highlight fringe
            cv2.line(img, (pt1[0] + 1, pt1[1] + 1), (pt2[0] + 1, pt2[1] + 1), halo_color, 1, cv2.LINE_AA)
            # Draw dark fracture line
            cv2.line(img, pt1, pt2, crack_color, 2, cv2.LINE_AA)

        # Calculate bounding box
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        pad = 6
        x1 = float(max(0, min(xs) - pad))
        y1 = float(max(0, min(ys) - pad))
        x2 = float(min(self.width, max(xs) + pad))
        y2 = float(min(self.height, max(ys) + pad))

        if (x2 - x1) < 8 or (y2 - y1) < 8:
            return None
        return [x1, y1, x2, y2]

    def _draw_scratch(self, img: np.ndarray) -> Optional[List[float]]:
        """Draw high-contrast directional scratch with shadowed edge."""
        margin = 30
        x1 = self.rng.randint(margin, self.width - margin)
        y1 = self.rng.randint(margin, self.height - margin)
        
        length = self.rng.uniform(30.0, 75.0)
        angle = self.rng.uniform(0, 2 * np.pi)
        
        x2 = int(np.clip(x1 + length * np.cos(angle), 5, self.width - 5))
        y2 = int(np.clip(y1 + length * np.sin(angle), 5, self.height - 5))

        bright_specular = (245, 245, 245)
        dark_trench = (30, 30, 30)

        # Scratch line with shadowed groove adjacent to bright groove
        cv2.line(img, (x1, y1), (x2, y2), dark_trench, 2, cv2.LINE_AA)
        cv2.line(img, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), bright_specular, 1, cv2.LINE_AA)

        pad = 6
        bx1 = float(max(0, min(x1, x2) - pad))
        by1 = float(max(0, min(y1, y2) - pad))
        bx2 = float(min(self.width, max(x1, x2) + pad))
        by2 = float(min(self.height, max(y1, y2) + pad))

        if (bx2 - bx1) < 6 or (by2 - by1) < 6:
            return None
        return [bx1, by1, bx2, by2]

    def _draw_pit(self, img: np.ndarray) -> Optional[List[float]]:
        """Draw localized corrosion pit / void crater with specular rim."""
        margin = 35
        cx = self.rng.randint(margin, self.width - margin)
        cy = self.rng.randint(margin, self.height - margin)
        
        axes_a = self.rng.randint(5, 14)
        axes_b = self.rng.randint(4, 12)
        rot_angle = self.rng.randint(0, 180)

        crater_shadow = (20, 20, 20)
        specular_rim = (240, 240, 240)

        # Specular rim on upper-left edge
        cv2.ellipse(img, (cx - 1, cy - 1), (axes_a + 2, axes_b + 2), rot_angle, 180, 360, specular_rim, 2, cv2.LINE_AA)
        # Deep dark pit core
        cv2.ellipse(img, (cx, cy), (axes_a, axes_b), rot_angle, 0, 360, crater_shadow, -1, cv2.LINE_AA)

        max_radius = max(axes_a, axes_b) + 6
        bx1 = float(max(0, cx - max_radius))
        by1 = float(max(0, cy - max_radius))
        bx2 = float(min(self.width, cx + max_radius))
        by2 = float(min(self.height, cy + max_radius))

        return [bx1, by1, bx2, by2]

    def apply_lighting_drift(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Simulate edge illumination drift (overexposure, underexposure, or lighting drop).

        Args:
            img: Input image [H, W, 3] uint8.
            factor: Luminance multiplier (e.g. 0.4 for severe underexposure, 1.8 for flare).
        """
        drifted = img.astype(np.float32) * factor
        return np.clip(drifted, 0, 255).astype(np.uint8)

    def apply_blur_drift(self, img: np.ndarray, kernel_size: int = 15, blur_type: str = "gaussian") -> np.ndarray:
        """Simulate optical defocus blur or conveyor belt mechanical vibration blur.

        Args:
            img: Input image [H, W, 3] uint8.
            kernel_size: Blur kernel size (must be odd).
            blur_type: 'gaussian' or 'motion'.
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if blur_type == "motion":
            # Horizontal motion blur kernel
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[int((k - 1) / 2), :] = np.ones(k, dtype=np.float32)
            kernel /= k
            return cv2.filter2D(img, -1, kernel)
        return cv2.GaussianBlur(img, (k, k), 0)

    def generate_frame(
        self,
        surface_type: str = "metal",
        defect_types: Optional[List[str]] = None,
        lighting_drift: Optional[float] = None,
        blur_drift: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate a single deterministic synthetic frame with defect annotations.

        Args:
            surface_type: 'metal' or 'pcb'.
            defect_types: List of defect names to draw (e.g. ['crack', 'pit']), or empty for clean surface.
            lighting_drift: Optional luminance scale factor.
            blur_drift: Optional blur kernel size.

        Returns:
            Tuple of (image [H, W, 3] BGR uint8, list of ground-truth defect annotations).
        """
        if surface_type == "pcb":
            img = self._generate_pcb_surface()
        else:
            img = self._generate_metal_surface()

        annotations: List[Dict[str, Any]] = []

        if defect_types:
            for defect_name in defect_types:
                bbox = None
                if defect_name == "crack":
                    bbox = self._draw_crack(img)
                elif defect_name == "scratch":
                    bbox = self._draw_scratch(img)
                elif defect_name == "pit":
                    bbox = self._draw_pit(img)

                if bbox is not None:
                    annotations.append({
                        "label": defect_name,
                        "class_id": CLASS_TO_ID[defect_name],
                        "bbox": bbox,
                        "confidence": 1.0,
                    })

        # Apply simulated optical degradation if requested
        if lighting_drift is not None:
            img = self.apply_lighting_drift(img, lighting_drift)

        if blur_drift is not None:
            img = self.apply_blur_drift(img, blur_drift)

        return img, annotations

    def generate_batch(
        self,
        num_frames: int = 200,
        defect_ratio: float = 0.5,
        seed: int = 42,
    ) -> List[Tuple[np.ndarray, List[Dict[str, Any]]]]:
        """Generate a deterministic batch of synthetic frames for benchmarks and evaluation.

        Args:
            num_frames: Total number of frames.
            defect_ratio: Ratio of frames containing defects vs clean surfaces.
            seed: Seed for deterministic reproducibility.

        Returns:
            List of (image, annotations) tuples.
        """
        self.set_seed(seed)
        dataset = []

        for i in range(num_frames):
            surface = "metal" if (i % 2 == 0) else "pcb"
            has_defect = (self.rng.uniform(0, 1) < defect_ratio)

            if has_defect:
                chosen_defect = self.rng.choice(DEFECT_CLASSES)
                defects = [str(chosen_defect)]
                # Occasionally insert multi-defect frame
                if self.rng.uniform(0, 1) < 0.2:
                    second_defect = self.rng.choice(DEFECT_CLASSES)
                    defects.append(str(second_defect))
            else:
                defects = []

            frame, annos = self.generate_frame(surface_type=surface, defect_types=defects)
            dataset.append((frame, annos))

        return dataset

    def generate_calibration_samples(self, num_samples: int = 60, seed: int = 999) -> List[np.ndarray]:
        """Generate calibrated input tensors for INT8 quantization calibration.

        Returns list of normalized float32 images of shape [1, 3, height, width].
        """
        self.set_seed(seed)
        tensors = []
        for i in range(num_samples):
            surface = "metal" if (i % 2 == 0) else "pcb"
            defects = [self.rng.choice(DEFECT_CLASSES)] if (i % 3 != 0) else []
            frame, _ = self.generate_frame(surface_type=surface, defect_types=defects)
            
            # Normalize to [0, 1], RGB, CHW
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))
            batched = np.expand_dims(chw, axis=0)
            tensors.append(batched)

        return tensors
