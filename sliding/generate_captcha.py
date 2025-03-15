import sys
import os
import random
import math
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def rotate(self, angles: Tuple[float, float, float]) -> "Point3D":
        rot = Rotation.from_euler("xyz", angles, degrees=True)
        point = rot.apply([self.x, self.y, self.z])
        return Point3D(point[0], point[1], point[2])

    def project(self, focal_length: float = 500) -> Tuple[float, float]:
        """Project 3D point to 2D plane"""
        if self.z + focal_length == 0:
            return (0, 0)
        factor = focal_length / (self.z + focal_length)
        return (self.x * factor, self.y * factor)


class Pattern:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 4), dtype=np.uint8)

    def squiggly_lines(self, num_lines: int = 15, complexity: float = 0.5):
        """Generate squiggly background lines"""
        for _ in range(num_lines):
            points = []
            x = 0
            while x < self.width:
                y = random.randint(0, self.height)
                points.append((x, y))
                x += random.randint(20, 50)

            if len(points) >= 2:
                points = np.array(points, np.int32).reshape((-1, 1, 2))

                color = (
                    random.randint(100, 200),
                    random.randint(100, 200),
                    random.randint(100, 200),
                    random.randint(100, 150),
                )

                cv2.polylines(
                    self.image,
                    [points],
                    False,
                    color,
                    thickness=random.randint(1, 3),
                    lineType=cv2.LINE_AA,
                )

    def crosshatch_pattern(self, spacing: int = 20, angle: float = 45):
        """Generate crosshatch pattern"""
        diagonal = math.sqrt(self.width**2 + self.height**2)

        for offset in range(-int(diagonal), int(diagonal), spacing):
            theta = math.radians(angle)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            x1 = offset * cos_t
            y1 = offset * sin_t
            x2 = x1 - diagonal * sin_t
            y2 = y1 + diagonal * cos_t

            color = (
                random.randint(50, 150),
                random.randint(50, 150),
                random.randint(50, 150),
                random.randint(50, 100),
            )

            cv2.line(
                self.image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def circular_pattern(self, num_circles: int = 20):
        """Generate concentric circles pattern"""
        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = math.sqrt(center_x**2 + center_y**2)

        for i in range(num_circles):
            radius = int((i + 1) * (max_radius / num_circles))

            color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 100),
            )

            cv2.circle(
                self.image,
                (center_x, center_y),
                radius,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def wave_pattern(self, num_waves: int = 10, amplitude: float = 50):
        """Generate wave pattern"""
        for wave in range(num_waves):
            points = []
            frequency = random.uniform(0.01, 0.05)
            phase = random.uniform(0, 2 * math.pi)

            for x in range(0, self.width, 2):
                y = self.height / 2 + amplitude * math.sin(frequency * x + phase)
                points.append((x, int(y)))

            if len(points) > 1:
                points = np.array(points, np.int32).reshape((-1, 1, 2))

                color = (
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(50, 100),
                )

                cv2.polylines(
                    self.image,
                    [points],
                    False,
                    color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

    def honeycomb_pattern(self, size: int = 30):
        """Generate honeycomb pattern"""

        def hex_corner(
            center: Tuple[float, float], size: float, i: int
        ) -> Tuple[float, float]:
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            return (
                center[0] + size * math.cos(angle_rad),
                center[1] + size * math.sin(angle_rad),
            )

        for y in range(-size, self.height + size, int(size * 1.5)):
            offset = 0 if (y // int(size * 1.5)) % 2 == 0 else size * math.sqrt(3) / 2
            for x in range(-size, self.width + size, int(size * math.sqrt(3))):
                corners = [hex_corner((x + offset, y), size, i) for i in range(6)]

                corners = np.array([(int(x), int(y)) for x, y in corners], np.int32)
                corners = corners.reshape((-1, 1, 2))

                color = (
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(50, 200),
                    random.randint(30, 70),
                )

                cv2.polylines(
                    self.image,
                    [corners],
                    True,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )


class Shape3D:
    def __init__(self, points: List[Point3D], edges: List[Tuple[int, int]]):
        self.points = points
        self.edges = edges
        self.rotation = [0, 0, 0]

    def rotate(self, angles: Tuple[float, float, float]):
        self.rotation = [r + a for r, a in zip(self.rotation, angles)]
        rotated_points = [p.rotate(self.rotation) for p in self.points]
        return rotated_points

    def project(self, focal_length: float = 500) -> List[Tuple[float, float]]:
        rotated = self.rotate((0, 0, 0))
        return [p.project(focal_length) for p in rotated]


class Cube(Shape3D):
    def __init__(self, center: Point3D, size: float):
        points = [
            Point3D(center.x - size / 2, center.y - size / 2, center.z - size / 2),
            Point3D(center.x + size / 2, center.y - size / 2, center.z - size / 2),
            Point3D(center.x + size / 2, center.y + size / 2, center.z - size / 2),
            Point3D(center.x - size / 2, center.y + size / 2, center.z - size / 2),
            Point3D(center.x - size / 2, center.y - size / 2, center.z + size / 2),
            Point3D(center.x + size / 2, center.y - size / 2, center.z + size / 2),
            Point3D(center.x + size / 2, center.y + size / 2, center.z + size / 2),
            Point3D(center.x - size / 2, center.y + size / 2, center.z + size / 2),
        ]
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        super().__init__(points, edges)


class Pyramid(Shape3D):
    def __init__(self, center: Point3D, base_size: float, height: float):
        points = [
            Point3D(
                center.x - base_size / 2,
                center.y + height / 2,
                center.z - base_size / 2,
            ),
            Point3D(
                center.x + base_size / 2,
                center.y + height / 2,
                center.z - base_size / 2,
            ),
            Point3D(
                center.x + base_size / 2,
                center.y + height / 2,
                center.z + base_size / 2,
            ),
            Point3D(
                center.x - base_size / 2,
                center.y + height / 2,
                center.z + base_size / 2,
            ),
            Point3D(center.x, center.y - height / 2, center.z),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
        super().__init__(points, edges)


class Sphere(Shape3D):
    def __init__(self, center: Point3D, radius: float, resolution: int = 10):
        points = []
        edges = []

        for i in range(resolution + 1):
            lat = math.pi * (-0.5 + float(i) / resolution)
            for j in range(resolution):
                lon = 2 * math.pi * float(j) / resolution
                x = center.x + radius * math.cos(lat) * math.cos(lon)
                y = center.y + radius * math.cos(lat) * math.sin(lon)
                z = center.z + radius * math.sin(lat)
                points.append(Point3D(x, y, z))

                if i < resolution:
                    edges.append((i * resolution + j, (i + 1) * resolution + j))
                if j < resolution - 1:
                    edges.append((i * resolution + j, i * resolution + j + 1))

        super().__init__(points, edges)


class Octahedron(Shape3D):
    def __init__(self, center: Point3D, size: float):
        points = [
            Point3D(center.x, center.y + size, center.z),
            Point3D(center.x + size, center.y, center.z),
            Point3D(center.x, center.y, center.z + size),
            Point3D(center.x - size, center.y, center.z),
            Point3D(center.x, center.y, center.z - size),
            Point3D(center.x, center.y - size, center.z),
        ]
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
        ]
        super().__init__(points, edges)


class Dodecahedron(Shape3D):
    def __init__(self, center: Point3D, size: float):
        phi = (1 + math.sqrt(5)) / 2
        points = []

        vertices = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
            (0, 1 / phi, phi),
            (0, 1 / phi, -phi),
            (0, -1 / phi, phi),
            (0, -1 / phi, -phi),
            (1 / phi, phi, 0),
            (-1 / phi, phi, 0),
            (1 / phi, -phi, 0),
            (-1 / phi, -phi, 0),
            (phi, 0, 1 / phi),
            (phi, 0, -1 / phi),
            (-phi, 0, 1 / phi),
            (-phi, 0, -1 / phi),
        ]

        for vertex in vertices:
            points.append(
                Point3D(
                    center.x + size * vertex[0],
                    center.y + size * vertex[1],
                    center.z + size * vertex[2],
                )
            )

        edges = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = math.sqrt(
                    (points[i].x - points[j].x) ** 2
                    + (points[i].y - points[j].y) ** 2
                    + (points[i].z - points[j].z) ** 2
                )
                if abs(dist - size * 2) < 0.1:
                    edges.append((i, j))

        super().__init__(points, edges)


class CaptchaGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        background_color: tuple[int, int, int] = (25, 20, 20),
    ):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.image[:] = background_color
        self.puzzle_piece = None
        self.puzzle_position = None

    def add_background_patterns(self):
        pattern = Pattern(self.width, self.height)
        pattern.squiggly_lines(num_lines=20, complexity=0.7)
        pattern.crosshatch_pattern(spacing=25, angle=30)
        pattern.circular_pattern(num_circles=15)
        pattern.wave_pattern(num_waves=8, amplitude=40)

        alpha_mask = pattern.image[:, :, 3] / 255.0

        for c in range(3):
            self.image[:, :, c] = (
                (1 - alpha_mask) * self.image[:, :, c]
                + alpha_mask * pattern.image[:, :, c]
            ).astype(np.uint8)

    def add_random_blur_circles(self, num_circles=5, min_radius=20, max_radius=60):
        """Add random circular blurred areas to confuse AI solvers"""
        for _ in range(num_circles):
            radius = random.randint(min_radius, max_radius)
            cx = random.randint(radius + 10, self.width - radius - 10)
            cy = random.randint(radius + 10, self.height - radius - 10)

            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            blur_radius = random.randint(4, 8) * 2 + 1

            roi = self.image.copy()

            blurred = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)

            mask_normalized = mask / 255.0
            mask_normalized = np.expand_dims(mask_normalized, axis=2)

            self.image = (
                blurred * mask_normalized + self.image * (1 - mask_normalized)
            ).astype(np.uint8)

    def add_noise_and_distortions(self):
        """Add various noise and distortions to confuse AI"""
        noise = np.random.rand(self.height, self.width)
        noise_mask = np.zeros((self.height, self.width), dtype=bool)
        noise_mask[noise < 0.01] = True

        noise_colors = np.random.randint(
            0, 255, (self.height, self.width, 3), dtype=np.uint8
        )

        for c in range(3):
            self.image[noise_mask, c] = noise_colors[noise_mask, c]

        for _ in range(20):
            start_x = random.randint(0, self.width - 1)
            start_y = random.randint(0, self.height - 1)
            end_x = random.randint(0, self.width - 1)
            end_y = random.randint(0, self.height - 1)
            color = (
                random.randint(30, 180),
                random.randint(30, 180),
                random.randint(30, 180),
            )
            cv2.line(
                self.image,
                (start_x, start_y),
                (end_x, end_y),
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def draw_3d_shape(self, shape: Shape3D, color: Tuple[int, int, int, int]):
        projected = shape.project()
        center_x = self.width // 2
        center_y = self.height // 2

        for edge in shape.edges:
            start = projected[edge[0]]
            end = projected[edge[1]]

            start_pos = (int(start[0] + center_x), int(start[1] + center_y))
            end_pos = (int(end[0] + center_x), int(end[1] + center_y))

            width = random.randint(1, 3)

            cv2.line(
                self.image,
                start_pos,
                end_pos,
                color[:3],
                thickness=width,
                lineType=cv2.LINE_AA,
            )

    def generate_puzzle_piece(self, min_size_ratio: float = 0.08, max_size_ratio: float = 0.20):
        """
        Generate puzzle piece shape and position with subtle blur effect and random shape
        
        Args:
            min_size_ratio: Minimum size of the puzzle piece as a ratio of the frame's smaller dimension
            max_size_ratio: Maximum size of the puzzle piece as a ratio of the frame's smaller dimension
        """
        smaller_dimension = min(self.width, self.height)
        min_size = int(smaller_dimension * min_size_ratio)
        max_size = int(smaller_dimension * max_size_ratio)
        
        min_size = max(40, min_size)
        max_size = min(max_size, smaller_dimension // 3)
        
        size = random.randint(min_size, max_size)
        
        margin_ratio = 0.05
        margin_x = int(self.width * margin_ratio)
        margin_y = int(self.height * margin_ratio)
        
        margin_x = max(size // 2, margin_x)
        margin_y = max(size // 2, margin_y)
        
        x = random.randint(margin_x, self.width - margin_x - size)
        y = random.randint(margin_y, self.height - margin_y - size)

        jitter_intensity = max(2, int(size * 0.05))
        jitter = lambda: random.randint(-jitter_intensity, jitter_intensity)

        piece_path = []

        point_factor = max(3, int(size / 30))
        top_points = random.randint(3, min(6, point_factor))
        right_points = random.randint(3, min(6, point_factor))

        has_top_bump = random.choice([True, False])
        has_right_bump = random.choice([True, False])
        has_bottom_bump = random.choice([True, False])
        has_left_bump = random.choice([True, False])

        bump_size = random.uniform(0.15, 0.3)

        top_edge = []
        for i in range(top_points):
            px = x + (i * size / (top_points - 1)) + jitter()
            py = y + jitter()

            if has_top_bump and i > 0 and i < top_points - 1:
                if random.random() < 0.7:
                    direction = -1 if random.random() < 0.5 else 1
                    py += direction * size * bump_size

            top_edge.append((int(px), int(py)))

        right_edge = []
        for i in range(right_points):
            px = x + size + jitter()
            py = y + (i * size / (right_points - 1)) + jitter()

            if has_right_bump and i > 0 and i < right_points - 1:
                if random.random() < 0.7:
                    direction = -1 if random.random() < 0.5 else 1
                    px += direction * size * bump_size

            right_edge.append((int(px), int(py)))

        bottom_edge = []
        bottom_points = random.randint(3, min(6, point_factor))
        for i in range(bottom_points):
            px = x + size - (i * size / (bottom_points - 1)) + jitter()
            py = y + size + jitter()

            if has_bottom_bump and i > 0 and i < bottom_points - 1:
                if random.random() < 0.7:
                    direction = -1 if random.random() < 0.5 else 1
                    py += direction * size * bump_size

            bottom_edge.append((int(px), int(py)))

        left_edge = []
        left_points = random.randint(3, min(6, point_factor))
        for i in range(left_points):
            px = x + jitter()
            py = y + size - (i * size / (left_points - 1)) + jitter()

            if has_left_bump and i > 0 and i < left_points - 1:
                if random.random() < 0.7:
                    direction = -1 if random.random() < 0.5 else 1
                    px += direction * size * bump_size

            left_edge.append((int(px), int(py)))

        piece_path = top_edge + right_edge + bottom_edge + left_edge

        self.puzzle_position = (x, y)
        self.puzzle_piece = piece_path

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        contour = np.array(piece_path, dtype=np.int32)
        cv2.fillPoly(mask, [contour], 255)

        original_content = self.image.copy()

        padding = max(5, int(size * 0.1))
        min_x = max(0, int(min(point[0] for point in piece_path) - padding))
        min_y = max(0, int(min(point[1] for point in piece_path) - padding))
        max_x = min(self.width, int(max(point[0] for point in piece_path) + padding))
        max_y = min(self.height, int(max(point[1] for point in piece_path) + padding))

        roi = self.image[min_y:max_y, min_x:max_x].copy()

        blur_radius = max(3, min(7, int(size / 20))) * 2 + 1
        blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)

        roi_mask = mask[min_y:max_y, min_x:max_x]

        roi_mask_3ch = np.stack([roi_mask, roi_mask, roi_mask], axis=2) / 255.0
        roi = (blurred_roi * roi_mask_3ch + roi * (1 - roi_mask_3ch)).astype(np.uint8)

        self.image[min_y:max_y, min_x:max_x] = roi

        min_x = int(min(point[0] for point in piece_path))
        min_y = int(min(point[1] for point in piece_path))
        max_x = int(max(point[0] for point in piece_path))
        max_y = int(max(point[1] for point in piece_path))

        padding = max(5, int(size * 0.1))
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.width, max_x + padding)
        max_y = min(self.height, max_y + padding)

        piece_width = max_x - min_x
        piece_height = max_y - min_y

        puzzle_piece = np.zeros((piece_height, piece_width, 4), dtype=np.uint8)

        puzzle_piece[:, :, :3] = original_content[min_y:max_y, min_x:max_x]
        puzzle_piece[:, :, 3] = mask[min_y:max_y, min_x:max_x]

        return puzzle_piece, mask, (min_x, min_y)

    def generate(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Generate the complete captcha image with puzzle piece"""
        self.add_background_patterns()

        shapes = [
            Cube(Point3D(0, 0, 0), 100),
            Pyramid(Point3D(-150, 0, 0), 80, 120),
            Sphere(Point3D(150, 0, 0), 60),
            Octahedron(Point3D(0, -150, 0), 70),
            Dodecahedron(Point3D(0, 150, 0), 50),
        ]

        num_shapes = random.randint(3, 5)
        selected_shapes = random.sample(shapes, num_shapes)

        for shape in selected_shapes:
            shape.rotate(
                (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
            )
            color = (
                random.randint(50, 150),
                random.randint(50, 150),
                random.randint(50, 150),
                random.randint(150, 200),
            )
            self.draw_3d_shape(shape, color)

        self.add_random_blur_circles(num_circles=random.randint(4, 7))
        self.add_noise_and_distortions()

        puzzle_piece, _, _ = self.generate_puzzle_piece()

        return self.image, puzzle_piece, self.puzzle_position


def create_captcha(
    width: int = 800, 
    height: int = 400, 
    background_color: tuple[int, int, int] = (25, 20, 20)
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Create a captcha image with puzzle piece verification
    
    Args:
        width: Width of the captcha image
        height: Height of the captcha image
        background_color: Background color in BGR format
        
    Returns:
        Tuple containing (captcha_image, puzzle_piece, puzzle_position)
    """
    generator = CaptchaGenerator(width, height, background_color)
    return generator.generate()


if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if len(sys.argv) > 2:
        width, height = int(sys.argv[1]), int(sys.argv[2])
    else:
        width, height = (800, 400)
    
    main_image, puzzle_piece, position = create_captcha(width, height)
    
    captcha_path = f"{output_dir}/captcha.png"
    puzzle_path = f"{output_dir}/puzzle_piece.png"

    cv2.imwrite(captcha_path, main_image)
    cv2.imwrite(puzzle_path, puzzle_piece)
    
    print(f"Generated CAPTCHA: {width}x{height}")
    print(f"Puzzle piece position: {position}")
