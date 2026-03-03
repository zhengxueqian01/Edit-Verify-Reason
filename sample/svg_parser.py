"""
SVG解析模块：解析SVG文件，提取坐标轴刻度和坐标映射
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re
try:
    import numpy as np
except ImportError:
    np = None


class SVGCoordinateMapper:
    """
    SVG坐标映射器：从SVG中提取坐标轴信息，建立数据坐标到图像坐标的映射
    """
    
    def __init__(self, svg_path: str):
        """
        初始化坐标映射器
        
        Args:
            svg_path: SVG文件路径
        """
        self.svg_path = Path(svg_path)
        if not self.svg_path.exists():
            raise FileNotFoundError(f"SVG文件不存在: {svg_path}")
        
        self.tree = ET.parse(self.svg_path)
        self.root = self.tree.getroot()
        
        # 读取原始SVG内容（用于提取注释）
        with open(self.svg_path, 'r', encoding='utf-8') as f:
            self.svg_content = f.read()
        
        # 命名空间
        self.ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # 解析结果
        self.axes_bounds: Optional[Dict[str, float]] = None
        # 注意：x_ticks和y_ticks中的坐标是SVG viewBox坐标系统中的坐标，不是PNG像素坐标
        self.x_ticks: List[Tuple[float, float]] = []  # [(svg_x, data_value), ...] SVG坐标系统中的x坐标和数据值
        self.y_ticks: List[Tuple[float, float]] = []  # [(svg_y, data_value), ...] SVG坐标系统中的y坐标和数据值
        self.x_range: Optional[Tuple[float, float]] = None  # (min, max) in data coordinates
        self.y_range: Optional[Tuple[float, float]] = None  # (min, max) in data coordinates
        # 注意：existing_points中的坐标是SVG viewBox坐标系统中的坐标，不是PNG像素坐标
        # 如果需要用于PNG绘制，需要使用convert_svg_coords_to_png函数进行转换
        self.existing_points: List[Tuple[float, float]] = []  # [(svg_x, svg_y), ...] 现有散点的SVG坐标
        
        # 点的大小信息（SVG坐标系统中的大小）
        self.point_size_svg: Optional[float] = None  # 点在SVG坐标系统中的直径
        self.point_marker: Optional[str] = None  # 点的标记样式
        
        # 解析SVG
        self._parse()
    
    def _parse(self):
        """解析SVG文件，提取坐标轴信息"""
        # 1. 找到axes区域（使用命名空间）
        # SVG使用默认命名空间，需要指定命名空间来查找元素
        axes = self.root.find('.//{http://www.w3.org/2000/svg}g[@id="axes_1"]')
        
        if axes is None:
            raise ValueError("未找到axes_1元素")
        
        # 2. 获取绘图区域边界（patch_2）
        patch = axes.find('.//{http://www.w3.org/2000/svg}g[@id="patch_2"]')
        if patch is not None:
            path = patch.find('.//{http://www.w3.org/2000/svg}path')
            if path is not None:
                d_attr = path.get('d', '')
                # 解析路径，提取边界
                # 格式: M x1 y1 L x2 y2 L x3 y3 L x4 y4 z
                coords = re.findall(r'[ML]\s+([\d.]+)\s+([\d.]+)', d_attr)
                if len(coords) >= 4:
                    x_coords = [float(c[0]) for c in coords]
                    y_coords = [float(c[1]) for c in coords]
                    self.axes_bounds = {
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords)
                    }
        
        # 3. 解析X轴刻度
        self._parse_x_ticks()
        
        # 4. 解析Y轴刻度
        self._parse_y_ticks()
        
        # 5. 提取现有散点位置
        self._extract_existing_points()
        
        # 6. 提取点的大小信息
        self._extract_point_size()
        
        # 7. 计算数据坐标范围
        if self.x_ticks and self.y_ticks:
            x_values = [t[1] for t in self.x_ticks]
            y_values = [t[1] for t in self.y_ticks]
            self.x_range = (min(x_values), max(x_values))
            self.y_range = (min(y_values), max(y_values))
        elif self.existing_points:
            # 如果无法从刻度获取范围，从现有点推断
            # 需要先将像素坐标转换为数据坐标
            if self.axes_bounds:
                pixel_points = np.array(self.existing_points)
                # 使用简化的线性映射（假设范围是0-100）
                # 实际应该从刻度推断
                self.x_range = (0, 100)
                self.y_range = (0, 100)
    
    def _parse_x_ticks(self):
        """解析X轴刻度，提取像素坐标和对应的数据值"""
        axis = self.root.find('.//{http://www.w3.org/2000/svg}g[@id="matplotlib.axis_1"]')
        if axis is None:
            return
        
        # 查找所有xtick（ElementTree不支持starts-with，需要手动过滤）
        all_g_elements = axis.findall('.//{http://www.w3.org/2000/svg}g')
        xticks = [g for g in all_g_elements if g.get('id', '').startswith('xtick_')]
        
        for xtick in xticks:
            # 获取刻度线的x位置（像素坐标）
            # 刻度线的实际位置在use元素的x属性中
            pixel_x = None
            
            # 首先尝试从use元素中提取x坐标
            use_elements = xtick.findall('.//{http://www.w3.org/2000/svg}use')
            for use_elem in use_elements:
                x_attr = use_elem.get('x')
                if x_attr:
                    try:
                        pixel_x = float(x_attr)
                        break
                    except ValueError:
                        pass
            
            # 如果没找到use元素，尝试从path的d属性中提取（向后兼容）
            if pixel_x is None:
                line = xtick.find('.//{http://www.w3.org/2000/svg}path[@d]')
                if line is not None:
                    d_attr = line.get('d', '')
                    # 格式: M x y L x y，提取第一个x坐标（竖直线，x坐标相同）
                    match = re.search(r'M\s+([\d.]+)', d_attr)
                    if match:
                        pixel_x = float(match.group(1))
            
            if pixel_x is not None:
                    
                    # 获取标签文本（数据值）
                    # 从原始SVG内容中提取注释（ElementTree不解析注释）
                    text_content = None
                    # 查找xtick下的text元素
                    all_text_g = xtick.findall('.//{http://www.w3.org/2000/svg}g')
                    text_elem = None
                    for g in all_text_g:
                        if g.get('id', '').startswith('text_'):
                            text_elem = g
                            break
                    if text_elem is not None:
                        text_id = text_elem.get('id', '')
                        if text_id:
                            # 在原始SVG内容中查找text元素及其后的注释
                            # 注释通常在text元素开始标签之后
                            pattern = rf'<g\s+id="{re.escape(text_id)}"[^>]*>\s*<!--\s*([\d.]+)\s*-->'
                            match = re.search(pattern, self.svg_content)
                            if match:
                                text_content = match.group(1)
                    
                    # 如果没找到，尝试在整个xtick范围内查找注释
                    if text_content is None:
                        xtick_id = xtick.get('id', '')
                        if xtick_id:
                            # 查找xtick元素内的注释
                            pattern = rf'<g\s+id="{re.escape(xtick_id)}"[^>]*>.*?<!--\s*([\d.]+)\s*-->'
                            match = re.search(pattern, self.svg_content, re.DOTALL)
                            if match:
                                text_content = match.group(1)
                    
                    # 如果找到了文本值
                    if text_content:
                        try:
                            data_value = float(text_content)
                            self.x_ticks.append((pixel_x, data_value))
                        except ValueError:
                            pass
        
        # 按数据值排序
        self.x_ticks.sort(key=lambda x: x[1])
        
        # 如果通过文本解析失败，尝试从位置推断
        if not self.x_ticks and self.axes_bounds:
            # 使用简化的方法：假设刻度是均匀分布的
            # 这需要知道实际的刻度值，我们可以从metadata获取，或者使用默认值
            pass
    
    def _parse_y_ticks(self):
        """解析Y轴刻度，提取像素坐标和对应的数据值"""
        axis = self.root.find('.//{http://www.w3.org/2000/svg}g[@id="matplotlib.axis_2"]')
        if axis is None:
            return
        
        # 查找所有ytick（ElementTree不支持starts-with，需要手动过滤）
        all_g_elements = axis.findall('.//{http://www.w3.org/2000/svg}g')
        yticks = [g for g in all_g_elements if g.get('id', '').startswith('ytick_')]
        
        for ytick in yticks:
            # 获取刻度线的y位置（像素坐标）
            # 刻度线的实际位置在use元素的y属性中
            pixel_y = None
            
            # 首先尝试从use元素中提取y坐标
            use_elements = ytick.findall('.//{http://www.w3.org/2000/svg}use')
            for use_elem in use_elements:
                y_attr = use_elem.get('y')
                if y_attr:
                    try:
                        pixel_y = float(y_attr)
                        break
                    except ValueError:
                        pass
            
            # 如果没找到use元素，尝试从path的d属性中提取（向后兼容）
            if pixel_y is None:
                line = ytick.find('.//{http://www.w3.org/2000/svg}path[@d]')
                if line is not None:
                    d_attr = line.get('d', '')
                    # 格式: M x y L x y，提取第一个y坐标（横线，y坐标相同）
                    match = re.search(r'M\s+[\d.]+\s+([\d.]+)', d_attr)
                    if match:
                        pixel_y = float(match.group(1))
            
            if pixel_y is not None:
                    
                    # 获取标签文本（数据值）
                    # 从原始SVG内容中提取注释（ElementTree不解析注释）
                    text_content = None
                    # 查找ytick下的text元素
                    all_text_g = ytick.findall('.//{http://www.w3.org/2000/svg}g')
                    text_elem = None
                    for g in all_text_g:
                        if g.get('id', '').startswith('text_'):
                            text_elem = g
                            break
                    if text_elem is not None:
                        text_id = text_elem.get('id', '')
                        if text_id:
                            # 在原始SVG内容中查找text元素及其后的注释
                            # 注释通常在text元素开始标签之后
                            pattern = rf'<g\s+id="{re.escape(text_id)}"[^>]*>\s*<!--\s*([\d.]+)\s*-->'
                            match = re.search(pattern, self.svg_content)
                            if match:
                                text_content = match.group(1)
                    
                    # 如果没找到，尝试在整个ytick范围内查找注释
                    if text_content is None:
                        ytick_id = ytick.get('id', '')
                        if ytick_id:
                            # 查找ytick元素内的注释
                            pattern = rf'<g\s+id="{re.escape(ytick_id)}"[^>]*>.*?<!--\s*([\d.]+)\s*-->'
                            match = re.search(pattern, self.svg_content, re.DOTALL)
                            if match:
                                text_content = match.group(1)
                    
                    # 如果找到了文本值
                    if text_content:
                        try:
                            data_value = float(text_content)
                            self.y_ticks.append((pixel_y, data_value))
                        except ValueError:
                            pass
        
        # 按数据值排序
        self.y_ticks.sort(key=lambda x: x[1])
    
    def _extract_existing_points(self):
        """
        从SVG中提取现有散点的坐标
        
        注意：提取的坐标是SVG viewBox坐标系统中的坐标，不是PNG像素坐标。
        这些坐标基于SVG的viewBox定义的用户坐标空间（例如0-432范围）。
        如果需要用于PNG绘制，需要使用tools.pixel_drawing.convert_svg_coords_to_png函数进行转换。
        """
        # 查找PathCollection_1元素（包含所有散点）
        axes = self.root.find('.//{http://www.w3.org/2000/svg}g[@id="axes_1"]')
        if axes is None:
            return
        
        # 首先尝试从PathCollection_1中提取
        path_collection = axes.find('.//{http://www.w3.org/2000/svg}g[@id="PathCollection_1"]')
        if path_collection is not None:
            # PathCollection_1下的每个g元素包含一个use元素，这些是散点
            # 查找所有use元素（使用命名空间）
            use_elements = path_collection.findall('.//{http://www.w3.org/2000/svg}use')
            
            for use_elem in use_elements:
                x_attr = use_elem.get('x')
                y_attr = use_elem.get('y')
                
                if x_attr and y_attr:
                    try:
                        pixel_x = float(x_attr)
                        pixel_y = float(y_attr)
                        self.existing_points.append((pixel_x, pixel_y))
                    except ValueError:
                        pass
        
        # 如果从PathCollection_1没找到，尝试在整个axes_1中查找所有use元素
        if not self.existing_points:
            use_elements = axes.findall('.//{http://www.w3.org/2000/svg}use')
            
            for use_elem in use_elements:
                x_attr = use_elem.get('x')
                y_attr = use_elem.get('y')
                
                if x_attr and y_attr:
                    try:
                        pixel_x = float(x_attr)
                        pixel_y = float(y_attr)
                        self.existing_points.append((pixel_x, pixel_y))
                    except ValueError:
                        pass
        
        # 如果还是没找到，尝试另一种方式（查找所有 use 元素，不管命名空间）
        if not self.existing_points:
            for elem in axes.iter():
                if elem.tag.endswith('use') or 'use' in elem.tag.lower():
                    x_attr = elem.get('x')
                    y_attr = elem.get('y')
                    if x_attr and y_attr:
                        try:
                            pixel_x = float(x_attr)
                            pixel_y = float(y_attr)
                            self.existing_points.append((pixel_x, pixel_y))
                        except ValueError:
                            pass
    
    def _extract_point_size(self):
        """从SVG中提取原始点的大小"""
        # 查找PathCollection_1元素
        axes = self.root.find('.//{http://www.w3.org/2000/svg}g[@id="axes_1"]')
        if axes is None:
            return
        
        path_collection = axes.find('.//{http://www.w3.org/2000/svg}g[@id="PathCollection_1"]')
        if path_collection is None:
            return
        
        # 查找第一个use元素，获取它引用的path定义
        use_elem = path_collection.find('.//{http://www.w3.org/2000/svg}use')
        if use_elem is None:
            return
        
        # 获取引用的path ID
        href = use_elem.get('{http://www.w3.org/1999/xlink}href')
        if not href or not href.startswith('#'):
            return
        
        path_id = href[1:]
        
        # 查找path定义
        path_def = self.root.find(f'.//{{http://www.w3.org/2000/svg}}path[@id="{path_id}"]')
        if path_def is None:
            return
        
        # 从path的d属性中提取点的大小
        d_attr = path_def.get('d', '')
        if not d_attr:
            return
        
        # 解析路径，提取所有坐标点
        # 路径格式通常是圆形：M 0 y1 C ... C ... z
        # 我们需要找到y的最大值和最小值来计算直径
        import re
        coords = re.findall(r'[-]?[\d.]+', d_attr)
        
        if len(coords) >= 2:
            # 尝试提取y坐标（通常是偶数索引）
            y_coords = []
            for i in range(1, len(coords), 2):  # 从索引1开始，每隔一个取一个（y坐标）
                try:
                    y_coords.append(float(coords[i]))
                except ValueError:
                    pass
            
            if y_coords:
                # 计算直径（最大y - 最小y）
                y_max = max(y_coords)
                y_min = min(y_coords)
                diameter = y_max - y_min
                
                # 如果直径合理（通常在5-20之间），使用它
                if 1.0 <= diameter <= 50.0:
                    self.point_size_svg = diameter
                    self.point_marker = 'circle'  # 默认是圆形
                    return
        
        # 如果无法从路径解析，尝试从style中提取
        style_attr = path_def.get('style', '')
        if 'marker' in style_attr.lower():
            if 'square' in style_attr.lower():
                self.point_marker = 'square'
            elif 'x' in style_attr.lower() or 'cross' in style_attr.lower():
                self.point_marker = 'x'
            else:
                self.point_marker = 'circle'
        
        # 如果还是无法确定，使用默认值（基于常见matplotlib点大小）
        if self.point_size_svg is None:
            # matplotlib默认点大小通常是6-10个点（在SVG坐标系统中）
            # 我们使用一个合理的默认值
            self.point_size_svg = 10.0
            self.point_marker = 'circle'
    
    def get_point_size_png(self, png_path: str) -> Optional[float]:
        """
        获取点在PNG像素坐标系统中的大小
        
        Args:
            png_path: PNG图片路径
            
        Returns:
            点在PNG像素坐标系统中的直径（像素），如果无法计算则返回None
        """
        if self.point_size_svg is None:
            return None
        
        try:
            from PIL import Image
            from tools.pixel_drawing import get_svg_dimensions
            
            # 获取SVG和PNG的尺寸
            svg_width, svg_height = get_svg_dimensions(str(self.svg_path))
            img = Image.open(png_path)
            png_width, png_height = img.size
            
            # 计算缩放比例（使用平均缩放比例）
            scale_x = png_width / svg_width
            scale_y = png_height / svg_height
            scale = (scale_x + scale_y) / 2.0  # 使用平均缩放比例
            
            # 将SVG坐标系统中的大小转换为PNG像素大小
            point_size_png = self.point_size_svg * scale
            
            return point_size_png
        except Exception as e:
            print(f"⚠️ 计算点大小时出错: {e}")
            return None
    
    def data_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """
        将数据坐标转换为SVG坐标系统中的坐标
        使用实际的刻度位置进行线性插值
        
        注意：返回的坐标是SVG viewBox坐标系统中的坐标，不是PNG像素坐标。
        如果需要用于PNG绘制，需要使用tools.pixel_drawing.convert_svg_coords_to_png函数进行转换。
        
        Args:
            x: 数据x坐标
            y: 数据y坐标
            
        Returns:
            (svg_x, svg_y) SVG坐标系统中的坐标（基于viewBox）
        """
        if not self.x_ticks or not self.y_ticks:
            raise ValueError("坐标映射未初始化，请先解析SVG刻度")
        
        # X轴映射：使用刻度点进行线性插值
        pixel_x = self._interpolate_axis(x, self.x_ticks, is_x=True)
        
        # Y轴映射：注意SVG的y轴是向下的，需要反转
        # 对于Y轴，数据值越大，像素坐标越小（因为SVG Y轴向下）
        # 所以我们需要反转y_ticks的像素坐标顺序
        y_ticks_reversed = [(pixel, data) for pixel, data in reversed(self.y_ticks)]
        pixel_y = self._interpolate_axis(y, y_ticks_reversed, is_x=False)
        
        return pixel_x, pixel_y
    
    def _interpolate_axis(self, data_value: float, ticks: List[Tuple[float, float]], is_x: bool) -> float:
        """
        在刻度点之间进行线性插值
        
        Args:
            data_value: 数据值
            ticks: 刻度列表 [(pixel_pos, data_value), ...]
            is_x: 是否是x轴
            
        Returns:
            像素坐标
        """
        if len(ticks) < 2:
            raise ValueError("至少需要2个刻度点才能进行插值")
        
        # 找到数据值所在的区间
        for i in range(len(ticks) - 1):
            pixel1, data1 = ticks[i]
            pixel2, data2 = ticks[i + 1]
            
            if data1 <= data_value <= data2:
                # 线性插值
                if data2 != data1:
                    ratio = (data_value - data1) / (data2 - data1)
                    pixel = pixel1 + ratio * (pixel2 - pixel1)
                else:
                    pixel = pixel1
                return pixel
        
        # 如果超出范围，进行外推
        if data_value < ticks[0][1]:
            # 使用前两个点外推
            pixel1, data1 = ticks[0]
            pixel2, data2 = ticks[1]
            if data2 != data1:
                ratio = (data_value - data1) / (data2 - data1)
                pixel = pixel1 + ratio * (pixel2 - pixel1)
            else:
                pixel = pixel1
        else:
            # 使用最后两个点外推
            pixel1, data1 = ticks[-2]
            pixel2, data2 = ticks[-1]
            if data2 != data1:
                ratio = (data_value - data1) / (data2 - data1)
                pixel = pixel1 + ratio * (pixel2 - pixel1)
            else:
                pixel = pixel2
        
        return pixel
    
    def pixel_to_data(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        将SVG坐标系统中的坐标转换为数据坐标
        
        注意：输入的坐标应该是SVG viewBox坐标系统中的坐标，不是PNG像素坐标。
        如果输入的是PNG像素坐标，需要先转换为SVG坐标（使用反向的convert_svg_coords_to_png转换）。
        
        Args:
            pixel_x: SVG坐标系统中的x坐标（基于viewBox）
            pixel_y: SVG坐标系统中的y坐标（基于viewBox）
            
        Returns:
            (data_x, data_y) 数据坐标
        """
        if not self.x_ticks or not self.y_ticks:
            raise ValueError("坐标刻度未初始化，无法进行像素到数据坐标的反向映射")

        def interpolate_pixel_to_data(pixel_value: float, ticks: List[Tuple[float, float]]) -> float:
            """
            在刻度点之间进行线性插值：已知 (pixel, data)，求给定 pixel 对应的 data
            
            为了兼容坐标轴正向/反向（比如Y轴向下），这里不假设 pixel 单调递增，
            而是按 pixel 排序后，在相邻区间内插值。
            """
            if len(ticks) < 2:
                raise ValueError("至少需要2个刻度点才能进行插值")
            
            # 按像素坐标排序，确保 pixel 单调
            ticks_sorted = sorted(ticks, key=lambda t: t[0])
            
            # 在刻度区间内查找
            for i in range(len(ticks_sorted) - 1):
                p1, d1 = ticks_sorted[i]
                p2, d2 = ticks_sorted[i + 1]
                
                # 允许 p1 > p2 的情况，统一处理为区间 [min_p, max_p]
                min_p, max_p = (p1, p2) if p1 <= p2 else (p2, p1)
                if min_p <= pixel_value <= max_p:
                    if p2 != p1:
                        ratio = (pixel_value - p1) / (p2 - p1)
                        return d1 + ratio * (d2 - d1)
                    else:
                        return d1
            
            # 如果超出范围，使用边界进行外推
            # 小于最小像素：使用前两个点
            p1, d1 = ticks_sorted[0]
            p2, d2 = ticks_sorted[1]
            if pixel_value < min(p1, p2):
                if p2 != p1:
                    ratio = (pixel_value - p1) / (p2 - p1)
                    return d1 + ratio * (d2 - d1)
                return d1
            
            # 大于最大像素：使用最后两个点
            p1, d1 = ticks_sorted[-2]
            p2, d2 = ticks_sorted[-1]
            if p2 != p1:
                ratio = (pixel_value - p1) / (p2 - p1)
                return d1 + ratio * (d2 - d1)
            return d2

        # 使用刻度信息进行反向映射，避免依赖 axes_bounds 导致方向错误
        data_x = interpolate_pixel_to_data(pixel_x, self.x_ticks)
        data_y = interpolate_pixel_to_data(pixel_y, self.y_ticks)
        
        return float(data_x), float(data_y)
    
    def get_existing_points_data_coords(self) -> List[Tuple[float, float]]:
        """
        获取现有散点的数据坐标
        
        Returns:
            现有散点的数据坐标列表 [(x1, y1), (x2, y2), ...]
        """
        if not self.existing_points:
            return []
        
        data_points = []
        for pixel_x, pixel_y in self.existing_points:
            try:
                data_x, data_y = self.pixel_to_data(pixel_x, pixel_y)
                data_points.append((data_x, data_y))
            except:
                # 如果转换失败，跳过这个点
                pass
        
        return data_points
    
    def get_mapping_formula(self) -> str:
        """
        获取坐标映射公式的字符串表示
        
        Returns:
            坐标映射公式的字符串
        """
        if not self.x_ticks or not self.y_ticks:
            return "坐标映射未初始化"
        
        formula_lines = []
        formula_lines.append("=" * 60)
        formula_lines.append("坐标映射公式")
        formula_lines.append("=" * 60)
        
        # X轴映射公式
        formula_lines.append("\n【X轴映射（数据坐标 → 像素坐标）】")
        formula_lines.append(f"刻度点: {self.x_ticks}")
        
        if len(self.x_ticks) >= 2:
            x_min_pixel, x_min_data = self.x_ticks[0]
            x_max_pixel, x_max_data = self.x_ticks[-1]
            
            if x_max_data != x_min_data:
                slope_x = (x_max_pixel - x_min_pixel) / (x_max_data - x_min_data)
                formula_lines.append(f"线性插值公式: pixel_x = {x_min_pixel:.2f} + (data_x - {x_min_data:.2f}) * {slope_x:.4f}")
                formula_lines.append(f"简化公式: pixel_x = {x_min_pixel:.2f} + {slope_x:.4f} * (data_x - {x_min_data:.2f})")
            else:
                formula_lines.append(f"常数映射: pixel_x = {x_min_pixel:.2f}")
        
        # Y轴映射公式（注意反转）
        formula_lines.append("\n【Y轴映射（数据坐标 → 像素坐标，已反转）】")
        formula_lines.append(f"原始刻度点: {self.y_ticks}")
        
        # 反转后的刻度点
        y_ticks_reversed = [(pixel, data) for pixel, data in reversed(self.y_ticks)]
        formula_lines.append(f"反转后刻度点: {y_ticks_reversed}")
        
        if len(y_ticks_reversed) >= 2:
            y_min_pixel, y_min_data = y_ticks_reversed[0]
            y_max_pixel, y_max_data = y_ticks_reversed[-1]
            
            if y_max_data != y_min_data:
                slope_y = (y_max_pixel - y_min_pixel) / (y_max_data - y_min_data)
                formula_lines.append(f"线性插值公式: pixel_y = {y_min_pixel:.2f} + (data_y - {y_min_data:.2f}) * {slope_y:.4f}")
                formula_lines.append(f"简化公式: pixel_y = {y_min_pixel:.2f} + {slope_y:.4f} * (data_y - {y_min_data:.2f})")
            else:
                formula_lines.append(f"常数映射: pixel_y = {y_min_pixel:.2f}")
        
        # 反向映射公式
        formula_lines.append("\n【反向映射（像素坐标 → 数据坐标）】")
        if self.axes_bounds and self.x_range and self.y_range:
            x_min_data, x_max_data = self.x_range
            y_min_data, y_max_data = self.y_range
            x_min_pixel = self.axes_bounds['x_min']
            x_max_pixel = self.axes_bounds['x_max']
            y_min_pixel = self.axes_bounds['y_max']  # 注意Y轴方向
            y_max_pixel = self.axes_bounds['y_min']
            
            if x_max_pixel != x_min_pixel:
                formula_lines.append(f"X轴反向: data_x = {x_min_data:.2f} + (pixel_x - {x_min_pixel:.2f}) / {x_max_pixel - x_min_pixel:.2f} * {x_max_data - x_min_data:.2f}")
            
            if y_min_pixel != y_max_pixel:
                formula_lines.append(f"Y轴反向: data_y = {y_min_data:.2f} + (pixel_y - {y_min_pixel:.2f}) / {y_min_pixel - y_max_pixel:.2f} * {y_max_data - y_min_data:.2f}")
        
        # 分段插值说明
        formula_lines.append("\n【分段线性插值说明】")
        formula_lines.append("对于不在刻度点上的值，使用相邻两个刻度点进行线性插值：")
        formula_lines.append("  pixel = pixel1 + (data_value - data1) / (data2 - data1) * (pixel2 - pixel1)")
        formula_lines.append("其中 pixel1, data1 和 pixel2, data2 是相邻的两个刻度点")
        
        formula_lines.append("\n" + "=" * 60)
        
        return "\n".join(formula_lines)
    
    def print_mapping_formula(self):
        """打印坐标映射公式"""
        print(self.get_mapping_formula())
    
    def get_mapping_info(self) -> Dict:
        """
        获取坐标映射信息
        
        Returns:
            包含映射信息的字典
        """
        # 输出映射规则
        mapping_rules = {
            'x_axis_mapping': {
                'ticks': self.x_ticks,
                'min_pixel': min([t[0] for t in self.x_ticks]) if self.x_ticks else None,
                'max_pixel': max([t[0] for t in self.x_ticks]) if self.x_ticks else None,
                'min_data': min([t[1] for t in self.x_ticks]) if self.x_ticks else None,
                'max_data': max([t[1] for t in self.x_ticks]) if self.x_ticks else None,
            },
            'y_axis_mapping': {
                'ticks': self.y_ticks,
                'min_pixel': min([t[0] for t in self.y_ticks]) if self.y_ticks else None,
                'max_pixel': max([t[0] for t in self.y_ticks]) if self.y_ticks else None,
                'min_data': min([t[1] for t in self.y_ticks]) if self.y_ticks else None,
                'max_data': max([t[1] for t in self.y_ticks]) if self.y_ticks else None,
            }
        }
        
        return {
            'mapping_source': 'svg',
            'svg_path': str(self.svg_path),  # 保存SVG路径，用于坐标转换
            'axes_bounds': self.axes_bounds,
            'x_ticks': self.x_ticks,
            'y_ticks': self.y_ticks,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'existing_points_pixel': self.existing_points,
            'existing_points_count': len(self.existing_points),
            'point_size_svg': self.point_size_svg,  # 点在SVG坐标系统中的大小
            'point_marker': self.point_marker,  # 点的标记样式
            'mapping_rules': mapping_rules
        }


def parse_svg_coordinates(svg_path: str, metadata: Optional[Dict] = None) -> SVGCoordinateMapper:
    """
    解析SVG文件，提取坐标映射
    
    Args:
        svg_path: SVG文件路径
        metadata: 可选的元数据（已废弃，保留以兼容旧代码）
        
    Returns:
        SVGCoordinateMapper实例
    """
    mapper = SVGCoordinateMapper(svg_path)
    
    # 如果解析的刻度为空，尝试从现有散点推断范围
    if not mapper.x_ticks or not mapper.y_ticks:
        if mapper.existing_points and np is not None:
            # 将像素坐标转换为数据坐标（需要先有坐标映射）
            # 如果还没有坐标映射，使用简化的方法
            if mapper.axes_bounds:
                # 从现有散点的SVG坐标推断数据范围
                # 注意：existing_points中的坐标是SVG viewBox坐标系统中的坐标
                pixel_points = np.array(mapper.existing_points)
                pixel_x_min, pixel_x_max = pixel_points[:, 0].min(), pixel_points[:, 0].max()
                pixel_y_min, pixel_y_max = pixel_points[:, 1].min(), pixel_points[:, 1].max()
                
                # 假设数据范围是0-100（matplotlib默认）
                # 或者从axes_bounds推断
                x_min_pixel = mapper.axes_bounds['x_min']
                x_max_pixel = mapper.axes_bounds['x_max']
                y_min_pixel = mapper.axes_bounds['y_max']  # 注意Y轴方向
                y_max_pixel = mapper.axes_bounds['y_min']
                
                # 使用线性映射推断数据范围
                # 假设matplotlib默认范围是0-100
                mapper.x_range = (0, 100)
                mapper.y_range = (0, 100)
                
                # 创建虚拟刻度
                mapper.x_ticks = [
                    (x_min_pixel, 0),
                    (x_max_pixel, 100)
                ]
                mapper.y_ticks = [
                    (y_min_pixel, 0),
                    (y_max_pixel, 100)
                ]
    
    return mapper
