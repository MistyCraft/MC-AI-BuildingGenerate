import numpy as np
import pyvista as pv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from schem_utils import build_block_array, decode_block_data, load_schem_data, save_schem_data
from transform_utils import mirror_block_array, mirror_palette_states, rotate_block_array, rotate_palette_states
from path_config import (
    DEFAULT_INPUT_SCHEM,
    DEFAULT_OUTPUT_SCHEM,
    SCHEM_DIR,
    STRUCTURE_NPY_FILE,
    BLOCK_DATA_FILE,
    METADATA_FILE,
    RAW_BLOCK_DATA_DEBUG_FILE,
    PREVIEW_IMAGE_FILE,
    ensure_project_dirs,
)


@dataclass
class RunOptions:
    """保存一次运行中由用户选择的处理选项。"""

    selected_schem: Path
    export_debug: bool
    rotate_degrees: int
    mirror_axis: str | None
    visualization_mode: str | None
    visualization_style: str | None = None


class MinecraftSchematicProcessor:
    """Minecraft 结构处理核心类"""
    
    BLOCK_TYPE_MAP = {
        "log": 0, "planks": 1, "stairs": 2, "slab": 3, "fence": 4,
        "glass_pane": 5, "door": 6, "functional": 7, "grass_block": 8, "air": 9
    }

    STAIR_SUBTYPES = {
        "oak_stairs": 0, "dark_oak_stairs": 1, "birch_stairs": 2, "spruce_stairs": 3
    }

    def __init__(self, work_dir="."):
        self.work_dir = Path(work_dir)
        self.palette = {}
        self.block_array = None
        self.structure_data = None
        self.raw_block_data = None
        self.id_to_name = {}
        self.id_to_color = {}

    # region 读取与导出
    def load_from_schem(self, schem_file):
        """从 .schem 文件加载数据"""
        _, self.palette, block_data, self.width, self.height, self.length = load_schem_data(schem_file)
        self.raw_block_data = list(block_data)
        self._refresh_palette_cache()

        decoded = decode_block_data(block_data)
        self.block_array = build_block_array(decoded, self.width, self.height, self.length)
        return self

    def print_structure_summary(self):
        """打印当前结构的基础摘要，便于确认加载结果。"""
        air_id = self.palette.get('minecraft:air')
        air_id_value = int(air_id) if air_id is not None else None
        total_blocks = self.width * self.height * self.length
        if air_id is None:
            non_air_blocks = total_blocks
        else:
            non_air_blocks = int(np.count_nonzero(self.block_array != air_id_value))

        print(f"📏 结构尺寸: {self.width} x {self.height} x {self.length}")
        print(f"🧩 Palette 大小: {len(self.palette)}")
        print(f"🌫️  air 方块 ID: {air_id_value if air_id_value is not None else '未定义'}") #这行别动，表情后面需要多加一个空格看上去才对齐
        print(f"🧱 非空气方块数: {non_air_blocks} / {total_blocks}")
        return self

    def get_non_air_block_count(self):
        air_id = self.palette.get('minecraft:air')
        air_id_value = int(air_id) if air_id is not None else None
        if air_id_value is None:
            return int(self.width * self.height * self.length)
        return int(np.count_nonzero(self.block_array != air_id_value))

    def save_to_schem(self, output_file):
        """保存为 .schem 文件"""
        save_schem_data(self.block_array, self.palette, self.width, self.height, self.length, self.work_dir / output_file)
        print(f"✅ 已保存 .schem 文件 {output_file}")
        return self

    def load_from_txt(self):
        """从文本文件加载数据"""
        meta_file = METADATA_FILE
        data_file = BLOCK_DATA_FILE
        
        # 读取元数据
        with open(meta_file, 'r') as f:
            self.width, self.height, self.length = map(
                self._parse_meta_int,
                f.readline().strip().split(',')
            )
            self.palette = {
                line.rsplit(',',1)[0]: self._parse_meta_int(line.rsplit(',',1)[1])
                for line in f
            }
        self._refresh_palette_cache()

        # 读取方块数据
        decoded = []
        with open(data_file, 'r') as f:
            for line in f:
                x, y, z, name = line.strip().split(',', 3)
                decoded.append(self.palette.get(name, 0))
        
        self._build_3d_array(decoded)
        return self

    def export_to_txt(self):
        """导出 metadata.txt 和 block_data.txt"""
        with open(METADATA_FILE, 'w', encoding='utf-8') as meta_file:
            meta_file.write(f"{self.width},{self.height},{self.length}\n")
            for block_name, block_id in self.palette.items():
                meta_file.write(f"{block_name},{int(block_id)}\n")

        with open(BLOCK_DATA_FILE, 'w', encoding='utf-8') as data_file:
            for y in range(self.height):
                for z in range(self.length):
                    for x in range(self.width):
                        block_id = self.block_array[y, z, x]
                        block_name = self._id_to_name(block_id)
                        data_file.write(f"{x},{y},{z},{block_name}\n")

        print(f"✅ 已导出方块数据 {BLOCK_DATA_FILE}")
        print(f"✅ 已导出元数据 {METADATA_FILE}")
        return self

    def export_debug_txt(self):
        """导出调试文本，包括原始 BlockData。"""
        self.export_to_txt()
        with open(RAW_BLOCK_DATA_DEBUG_FILE, 'w', encoding='utf-8') as debug_file:
            for value in self.raw_block_data:
                debug_file.write(f"{value}\n")
        print(f"✅ 已导出原始 BlockData 调试文件 {RAW_BLOCK_DATA_DEBUG_FILE}")
        return self

    def verify_saved_schem(self, schem_file):
        """重新读取已保存的 .schem，并打印摘要用于回读校验。"""
        print(f"🔁 开始回读校验 {schem_file}")
        reloaded = MinecraftSchematicProcessor(self.work_dir).load_from_schem(schem_file)
        reloaded.print_structure_summary()
        print("✅ 回读校验完成")
        return self

    # endregion

    # region 摘要与校验
    def generate_npy(self, output_file):
        """生成 NPY 结构数据"""
        structure = np.zeros((self.width, self.height, self.length, 7), dtype=np.int16)
        
        for y in range(self.height):
            for z in range(self.length):
                for x in range(self.width):
                    block_id = self.block_array[y, z, x]
                    block_info = self._parse_block(self._id_to_name(block_id))
                    structure[x, y, z] = [block_id, *block_info]
        
        np.save(self.work_dir / output_file, structure)
        self.structure_data = structure
        print(f"✅ 已生成 NPY 数据 {output_file}")
        return self

    def validate_data(self, npy_file):
        """验证数据一致性"""
        structure = np.load(npy_file)
        expected_shape = (self.width, self.height, self.length)
        actual_shape = structure.shape[:3]
        if actual_shape != expected_shape:
            print(f"😭 NPY 结构尺寸不一致: 期望 {expected_shape}，实际 {actual_shape}")
            return self

        rebuild_array = structure[:, :, :, 0].astype(int)
        expected_array = np.transpose(self.block_array, (2, 0, 1))

        diff_positions = np.argwhere(expected_array != rebuild_array)
        if diff_positions.size == 0:
            print("✅ 数据一致性检查通过")
            return self

        first_diff = diff_positions[0]
        x, y, z = map(int, first_diff)
        expected_id = int(expected_array[x, y, z])
        actual_id = int(rebuild_array[x, y, z])
        print(f"😭 数据一致性检查发现 {len(diff_positions)} 处差异")
        print(f"🔎 首个差异位置: (x={x}, y={y}, z={z})")
        print(f"📤 当前内存方块 ID: {expected_id}")
        print(f"📥 NPY 还原方块 ID: {actual_id}")
        return self

    # endregion

    # region 可视化
    def visualize_3d(self, style='cube', mode='window', output_file=None):
        """3D 可视化。默认使用体素立方体预览，避免点云难以判断结构。"""
        start_time = time.perf_counter()
        air_id = self.palette.get('minecraft:air')
        air_id_value = int(air_id) if air_id is not None else None
        mask = self._build_visualization_mask(style, air_id_value)
        coords = np.argwhere(mask)
        if coords.size == 0:
            print("😭 没有可预览的非空气方块")
            return self

        points = np.column_stack((coords[:, 2], coords[:, 1], coords[:, 0])).astype(np.float32)
        block_ids = self.block_array[coords[:, 0], coords[:, 1], coords[:, 2]]
        colors = np.array([self.id_to_color.get(int(block_id), (1.0, 1.0, 1.0)) for block_id in block_ids], dtype=np.float32)

        if style == 'cube' and len(points) > 80000:
            print(f"⚠️ 当前预览方块数 {len(points)}，这种模式可能较慢")
        if style == 'shell':
            print(f"🪶 外壳预览方块数: {len(points)}")

        cloud = pv.PolyData(points)
        cloud['colors'] = colors

        plotter = pv.Plotter(off_screen=(mode == 'screenshot'))
        if style == 'point':
            plotter.add_points(cloud, rgb=True, point_size=18)
        else:
            cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0)
            voxel_mesh = cloud.glyph(geom=cube, scale=False, orient=False)
            plotter.add_mesh(voxel_mesh, scalars='colors', rgb=True, show_edges=False)

        # 旧版点云预览，先保留供对比，不删除。
        # if style == 'voxel':
        #     plotter.add_volume(cloud, shade=True, opacity='linear')
        # else:
        #     plotter.add_points(cloud, rgb=True, point_size=50)

        if mode == 'screenshot':
            target_file = output_file or PREVIEW_IMAGE_FILE
            plotter.show(screenshot=str(target_file), auto_close=True)
            print(f"🖼️ 已导出结构预览图 {target_file}")
            print(f"⏱️ 可视化耗时: {time.perf_counter() - start_time:.2f} 秒")
            return self

        if mode == 'window':
            self._bind_flight_controls(plotter, points)
            print("🎮 视角控制: W/S 前后, A/D 左右, Z/E 上下, R 重置视角")
            plotter.show(auto_close=True)
        print(f"⏱️ 可视化耗时: {time.perf_counter() - start_time:.2f} 秒")
        return self

    def _build_visualization_mask(self, style, air_id):
        solid_mask = self.block_array != air_id if air_id is not None else np.ones_like(self.block_array, dtype=bool)
        if style != 'shell':
            return solid_mask

        padded = np.pad(solid_mask, 1, mode='constant', constant_values=False)
        neighbor_all_solid = (
            padded[1:-1, 1:-1, :-2] &
            padded[1:-1, 1:-1, 2:] &
            padded[1:-1, :-2, 1:-1] &
            padded[1:-1, 2:, 1:-1] &
            padded[:-2, 1:-1, 1:-1] &
            padded[2:, 1:-1, 1:-1]
        )
        return solid_mask & ~neighbor_all_solid

    def _bind_flight_controls(self, plotter, points):
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
        center = (bounds_min + bounds_max) / 2.0
        span = float(np.max(bounds_max - bounds_min))
        step = max(span / 20.0, 1.0)
        far_clip = max(span * 8.0, 100.0)

        def move_camera(delta):
            camera = plotter.camera
            position = np.array(camera.position)
            focal_point = np.array(camera.focal_point)
            camera.position = tuple(position + delta)
            camera.focal_point = tuple(focal_point + delta)
            camera.clipping_range = (0.05, far_clip)
            plotter.reset_camera_clipping_range()
            plotter.render()

        def move_forward(scale):
            camera = plotter.camera
            direction = np.array(camera.focal_point) - np.array(camera.position)
            norm = np.linalg.norm(direction)
            if norm == 0:
                return
            move_camera((direction / norm) * step * scale)

        def move_right(scale):
            camera = plotter.camera
            direction = np.array(camera.focal_point) - np.array(camera.position)
            view_up = np.array(camera.up)
            right = np.cross(direction, view_up)
            norm = np.linalg.norm(right)
            if norm == 0:
                return
            move_camera((right / norm) * step * scale)

        def move_up(scale):
            camera = plotter.camera
            view_up = np.array(camera.up)
            norm = np.linalg.norm(view_up)
            if norm == 0:
                return
            move_camera((view_up / norm) * step * scale)

        def reset_camera():
            plotter.camera_position = [
                (center[0] + span * 1.5, center[1] + span * 1.2, center[2] + span * 1.5),
                tuple(center),
                (0.0, 0.0, 1.0),
            ]
            plotter.camera.clipping_range = (0.05, far_clip)
            plotter.reset_camera_clipping_range()
            plotter.render()

        plotter.add_key_event("w", lambda: move_forward(1.0))
        plotter.add_key_event("s", lambda: move_forward(-1.0))
        plotter.add_key_event("a", lambda: move_right(-1.0))
        plotter.add_key_event("d", lambda: move_right(1.0))
        plotter.add_key_event("z", lambda: move_up(-1.0))
        plotter.add_key_event("e", lambda: move_up(1.0))
        plotter.add_key_event("r", reset_camera)
        reset_camera()

    # endregion

    # region 结构变换
    def rotate_structure(self, degrees=90):
        """旋转结构"""
        normalized_degrees = degrees % 360
        if normalized_degrees == 0:
            print("📦 旋转角度为 0，跳过旋转")
            return self

        # block_array 的轴顺序是 (y, z, x)，绕 y 轴旋转应在 (z, x) 平面上旋转。
        steps = normalized_degrees // 90
        self.block_array = rotate_block_array(self.block_array, normalized_degrees)
        self.palette, self.block_array = rotate_palette_states(self.palette, self.block_array, normalized_degrees)
        self._refresh_palette_cache()

        if steps % 2 == 1:
            self.width, self.length = self.length, self.width
        print(f"🔄 结构已旋转 {degrees} 度")
        return self

    def mirror_structure(self, axis='x'):
        """镜像结构"""
        self.block_array = mirror_block_array(self.block_array, axis)
        self.palette, self.block_array = mirror_palette_states(self.palette, self.block_array, axis)
        self._refresh_palette_cache()
        print(f"🪞 结构已沿 {axis} 轴镜像")
        return self

    # endregion

    # region 内部辅助
    def _id_to_name(self, block_id):
        """将方块 ID 转换为名称"""
        return self.id_to_name.get(int(block_id), 'unknown')

    def _refresh_palette_cache(self):
        self.id_to_name = {int(block_id): block_name for block_name, block_id in self.palette.items()}
        self.id_to_color = {}
        for block_id, block_name in self.id_to_name.items():
            color_hash = hash(block_name) % 0xFFFFFF
            self.id_to_color[block_id] = (
                ((color_hash >> 16) & 0xFF) / 255.0,
                ((color_hash >> 8) & 0xFF) / 255.0,
                (color_hash & 0xFF) / 255.0,
            )

    def _parse_meta_int(self, value):
        value = str(value).strip()
        if '(' in value and value.endswith(')'):
            value = value.split('(', 1)[1][:-1]
        return int(value)

    def _parse_block(self, block_str):
        """解析方块属性"""
        match = re.match(r"minecraft:(\w+)(?:\[(.*?)\])?", block_str)
        if not match:
            return [-1]*6
        
        name, props = match.groups()
        prop_dict = dict(p.split('=') for p in props.split(',')) if props else {}
        
        # 属性解析逻辑
        attrs = []
        if 'stairs' in name:
            attrs.extend([
                ['north','east','south','west'].index(prop_dict.get('facing','north')),
                0 if prop_dict.get('half','bottom') == 'bottom' else 1,
                ['straight','inner_left','inner_right','outer_left','outer_right'].index(
                    prop_dict.get('shape','straight')),
            ])
        elif 'log' in name:
            attrs.append({'x':0, 'y':1, 'z':2}.get(prop_dict.get('axis','y'), 1))
        # 其他方块属性解析可在这里继续扩展
        
        return attrs + [-1]*(6-len(attrs))
    # endregion


# region 运行环境
def configure_console_output():
    """尽量将终端输出切换到 UTF-8，减少中文和 emoji 乱码。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
# endregion


# region 交互输入
def select_schem_file():
    """列出可用的 .schem 文件，并让用户按序号选择。"""
    schem_files = sorted(SCHEM_DIR.glob("*.schem"))
    if not schem_files:
        raise FileNotFoundError(f"未在目录中找到 .schem 文件: {SCHEM_DIR}")

    print("📂 检测到以下 .schem 文件：")
    for index, schem_file in enumerate(schem_files, start=1):
        default_mark = " ✅ 默认" if schem_file == DEFAULT_INPUT_SCHEM else ""
        print(f"{index}. {schem_file.name}{default_mark}")

    choice = input(f"请输入序号选择文件 [1-{len(schem_files)}]，直接回车使用默认文件：").strip()
    if not choice:
        print(f"✅ 已使用默认文件 {DEFAULT_INPUT_SCHEM.name}")
        return DEFAULT_INPUT_SCHEM

    if choice.isdigit():
        selected_index = int(choice)
        if 1 <= selected_index <= len(schem_files):
            selected_file = schem_files[selected_index - 1]
            print(f"📌 已选择文件 {selected_file.name}")
            return selected_file

    print(f"😭 输入无效，已回退到默认文件 {DEFAULT_INPUT_SCHEM.name}")
    return DEFAULT_INPUT_SCHEM


def ask_export_debug():
    """询问是否导出调试文本。"""
    choice = input("📝 是否导出调试 txt 文件？(y/n，直接回车默认 n)：").strip().lower()
    if choice in {"y", "yes"}:
        print("✅ 将导出调试 txt 文件")
        return True
    if choice in {"n", "no", ""}:
        print("📦 跳过调试 txt 导出")
        return False

    print("😭 输入无效，默认不导出调试 txt")
    return False


def ask_rotate_degrees():
    """询问是否执行旋转，以及旋转角度。"""
    choice = input("🔄 是否旋转结构？输入 0/90/180/270，直接回车默认 0：").strip()
    if choice == "":
        print("📦 跳过旋转")
        return 0

    if choice in {"0", "90", "180", "270"}:
        degrees = int(choice)
        if degrees == 0:
            print("📦 跳过旋转")
        else:
            print(f"✅ 将旋转 {degrees} 度")
        return degrees

    print("😭 输入无效，默认不旋转")
    return 0


def ask_mirror_axis():
    """询问是否执行镜像，以及镜像轴。"""
    choice = input("🪞 是否镜像结构？输入 x/z，直接回车默认不镜像：").strip().lower()
    if choice == "":
        print("📦 跳过镜像")
        return None

    if choice in {"x", "z"}:
        print(f"✅ 将沿 {choice} 轴镜像")
        return choice

    print("😭 输入无效，默认不镜像")
    return None


def ask_visualization_mode():
    """询问可视化模式。默认跳过，避免窗口阻塞主流程。"""
    choice = input("👀 是否预览结构？输入 w=窗口 / s=截图 / 直接回车跳过：").strip().lower()
    if choice == "":
        print("📦 跳过结构预览")
        return None
    if choice == "w":
        print("🪟 将打开预览窗口")
        return "window"
    if choice == "s":
        print(f"🖼️ 将导出预览截图 {PREVIEW_IMAGE_FILE}")
        return "screenshot"

    print("😭 输入无效，默认跳过结构预览")
    return None


def ask_visualization_style(non_air_blocks):
    """仅在需要预览时询问可视化风格。"""
    options = {
        "1": ("shell", "外壳体素"),
        "2": ("cube", "立方体体素"),
        "3": ("point", "纯点阵"),
    }
    if non_air_blocks > 80000:
        print(f"⚠️ 当前非空气方块数 {non_air_blocks}，推荐优先使用 1 或 3")
    print("🎨 请选择可视化风格：")
    print("1. 外壳体素")
    print("2. 立方体体素")
    print("3. 纯点阵")
    choice = input("请输入序号，直接回车默认 1：").strip()
    if choice == "":
        print("✅ 已选择可视化风格：外壳体素")
        return "shell"
    if choice in options:
        style, label = options[choice]
        print(f"✅ 已选择可视化风格：{label}")
        return style

    print("😭 输入无效，默认使用外壳体素")
    return "shell"
# endregion


# region 流程编排
def collect_run_options():
    """集中收集用户输入，避免主程序中混杂交互和数据处理逻辑。"""
    return RunOptions(
        selected_schem=select_schem_file(),
        export_debug=ask_export_debug(),
        rotate_degrees=ask_rotate_degrees(),
        mirror_axis=ask_mirror_axis(),
        visualization_mode=ask_visualization_mode(),
    )


def run_pipeline(options):
    """按照运行选项依次执行加载、预览、变换、保存和校验。"""
    processor = (
        MinecraftSchematicProcessor()
        .load_from_schem(options.selected_schem)
        .print_structure_summary()
    )

    if options.visualization_mode:
        options.visualization_style = ask_visualization_style(
            processor.get_non_air_block_count()
        )
    if options.export_debug:
        processor.export_debug_txt()
    if options.visualization_mode:
        processor.visualize_3d(
            style=options.visualization_style,
            mode=options.visualization_mode,
        )
    if options.rotate_degrees:
        processor.rotate_structure(options.rotate_degrees).print_structure_summary()
    if options.mirror_axis:
        processor.mirror_structure(options.mirror_axis).print_structure_summary()

    return (
        processor
        .save_to_schem(DEFAULT_OUTPUT_SCHEM)
        .verify_saved_schem(DEFAULT_OUTPUT_SCHEM)
        .generate_npy(STRUCTURE_NPY_FILE)
        .validate_data(STRUCTURE_NPY_FILE)
    )
# endregion


# 主程序运行流程：
# 1. 设置终端编码，并创建项目所需的输入、输出和训练数据目录。
# 2. 收集用户选项：输入结构、调试文本、旋转、镜像和预览方式。
# 3. 加载 .schem 文件，解析调色板和方块数组，并打印结构摘要。
# 4. 按需导出调试文本或预览原始结构，方便先确认输入是否正确。
# 5. 按用户选择依次执行旋转和镜像，并在每次变换后打印新摘要。
# 6. 保存变换后的 .schem，再重新读取一次，确认输出文件可以正常解析。
# 7. 生成供训练使用的 NPY 文件，并检查其尺寸和方块 ID 是否一致。
# 8. 打印整个处理流程的总耗时。
if __name__ == "__main__":
    total_start_time = time.perf_counter()
    configure_console_output()
    ensure_project_dirs()
    run_options = collect_run_options()
    run_pipeline(run_options)
    print(f"⏱️ 总流程耗时: {time.perf_counter() - total_start_time:.2f} 秒")
