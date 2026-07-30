import numpy as np
from nbtlib import Int

from schem_utils import join_block_state, split_block_state


def rotate_block_array(block_array, degrees):
    """绕 y 轴旋转方块数组；数组轴顺序为 (y, z, x)。"""
    steps = (degrees % 360) // 90
    return np.rot90(block_array, k=steps, axes=(1, 2))


def mirror_block_array(block_array, axis):
    """沿 Minecraft x 或 z 轴镜像方块数组。"""
    array_axis = 2 if axis == "x" else 1
    return np.flip(block_array, axis=array_axis)


def rotate_palette_states(palette, block_array, degrees):
    """旋转带方向属性的方块状态，并重建调色板 ID。"""
    steps = (degrees % 360) // 90
    if steps == 0:
        return palette, block_array

    facing_order = ["north", "east", "south", "west"]
    fence_rotate_90 = {
        "east": "north",
        "north": "west",
        "west": "south",
        "south": "east",
    }

    transformed_names = {}
    for block_name, block_id in palette.items():
        base_name, properties = split_block_state(block_name)

        if properties.get("facing") in facing_order:
            facing_index = facing_order.index(properties["facing"])
            properties["facing"] = facing_order[(facing_index - steps) % 4]

        if properties.get("axis") in {"x", "z"} and steps % 2 == 1:
            properties["axis"] = "z" if properties["axis"] == "x" else "x"

        if "fence" in base_name or "glass_pane" in base_name:
            for _ in range(steps):
                properties = remap_properties(properties, fence_rotate_90)

        transformed_names[int(block_id)] = join_block_state(base_name, properties)

    return rebuild_palette(transformed_names, block_array)


def mirror_palette_states(palette, block_array, axis):
    """镜像带方向属性的方块状态，并重建调色板 ID。"""
    mirror_direction = (
        {"north": "north", "south": "south", "east": "west", "west": "east"}
        if axis == "x"
        else {"north": "south", "south": "north", "east": "east", "west": "west"}
    )

    transformed_names = {}
    for block_name, block_id in palette.items():
        base_name, properties = split_block_state(block_name)

        if properties.get("facing") in mirror_direction:
            properties["facing"] = mirror_direction[properties["facing"]]

        if "door" in base_name and properties.get("hinge") in {"left", "right"}:
            properties["hinge"] = "right" if properties["hinge"] == "left" else "left"

        if "stairs" in base_name and properties.get("shape") in {
            "inner_left",
            "inner_right",
            "outer_left",
            "outer_right",
        }:
            properties["shape"] = (
                properties["shape"].replace("_left", "_right")
                if properties["shape"].endswith("_left")
                else properties["shape"].replace("_right", "_left")
            )

        if "fence" in base_name or "glass_pane" in base_name:
            properties = remap_properties(properties, mirror_direction)

        transformed_names[int(block_id)] = join_block_state(base_name, properties)

    return rebuild_palette(transformed_names, block_array)


def remap_properties(properties, direction_map):
    """按照方向映射重命名 north/east/south/west 属性。"""
    remapped = dict(properties)
    directional_values = {
        direction_map[key]: value
        for key, value in properties.items()
        if key in direction_map
    }
    for key in direction_map:
        remapped.pop(key, None)
    remapped.update(directional_values)
    return remapped


def rebuild_palette(transformed_names, block_array):
    """合并相同状态并返回连续 ID 的调色板和方块数组。"""
    name_to_new_id = {}
    old_to_new_id = {}
    for old_id in sorted(transformed_names):
        block_name = transformed_names[old_id]
        if block_name not in name_to_new_id:
            name_to_new_id[block_name] = len(name_to_new_id)
        old_to_new_id[old_id] = name_to_new_id[block_name]

    new_palette = {name: Int(block_id) for name, block_id in name_to_new_id.items()}
    new_block_array = remap_block_array(block_array, old_to_new_id)
    return new_palette, new_block_array


def remap_block_array(block_array, id_mapping):
    """按 ID 映射表批量替换方块数组。"""
    remap = np.vectorize(lambda block_id: id_mapping[int(block_id)], otypes=[np.int32])
    return remap(block_array)
