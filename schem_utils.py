import re

import numpy as np
from nbtlib import ByteArray, Compound, File, Int, load


def load_schem_data(schem_file):
    """读取 Sponge .schem 文件中的调色板、方块数据和尺寸。"""
    nbt_file = load(schem_file)
    # nbtlib 新版的 File 本身就是根映射，旧版则通过 root 属性访问。
    root = getattr(nbt_file, "root", nbt_file)
    palette = root["Palette"]
    block_data = root["BlockData"]
    width = int(root["Width"])
    height = int(root["Height"])
    length = int(root["Length"])
    return root, palette, block_data, width, height, length


def decode_block_data(data):
    """将 VarInt 编码的 BlockData 解码为方块 ID 列表。"""
    decoded = []
    value = 0
    shift = 0
    for byte in data:
        byte_value = int(byte) & 0xFF
        value |= (byte_value & 0x7F) << shift
        if byte_value & 0x80:
            shift += 7
            continue
        decoded.append(value)
        value = 0
        shift = 0

    if shift != 0:
        raise ValueError("BlockData 末尾存在不完整的 VarInt")
    return decoded


def encode_block_data(data):
    """将方块 ID 序列编码为 Sponge .schem 使用的 VarInt 字节流。"""
    encoded = []
    for raw_value in data:
        value = int(raw_value)
        while True:
            byte_value = value & 0x7F
            value >>= 7
            if value:
                byte_value |= 0x80
            encoded.append(byte_value if byte_value < 128 else byte_value - 256)
            if not value:
                break
    return encoded


def build_block_array(decoded, width, height, length):
    """将 y-z-x 顺序的一维方块 ID 列表还原为三维数组。"""
    expected_size = width * height * length
    if len(decoded) != expected_size:
        raise ValueError(f"BlockData 长度不匹配: 期望 {expected_size}，实际 {len(decoded)}")

    block_array = np.zeros((height, length, width), dtype=np.int32)
    index = 0
    for y in range(height):
        for z in range(length):
            for x in range(width):
                block_array[y, z, x] = decoded[index]
                index += 1
    return block_array


def save_schem_data(block_array, palette, width, height, length, output_file):
    """将当前结构写入 Sponge .schem 文件。"""
    flattened = block_array.reshape(-1)
    encoded = encode_block_data(flattened)
    root = Compound(
        {
            "Version": Int(2),
            "DataVersion": Int(2586),
            "Width": Int(width),
            "Height": Int(height),
            "Length": Int(length),
            "PaletteMax": Int(len(palette)),
            "Palette": Compound({name: Int(int(block_id)) for name, block_id in palette.items()}),
            "BlockData": ByteArray(encoded),
        }
    )
    File(root).save(output_file, gzipped=True)


def split_block_state(block_name):
    """把方块状态拆成基础名称和属性字典。"""
    match = re.match(r"^(minecraft:\w+)(?:\[(.*?)\])?$", block_name)
    if not match:
        return block_name, {}

    base_name, properties_text = match.groups()
    properties = {}
    if properties_text:
        properties = dict(item.split("=", 1) for item in properties_text.split(","))
    return base_name, properties


def join_block_state(base_name, properties):
    """将基础名称和属性字典重新组合成方块状态字符串。"""
    if not properties:
        return base_name
    properties_text = ",".join(f"{key}={value}" for key, value in properties.items())
    return f"{base_name}[{properties_text}]"
