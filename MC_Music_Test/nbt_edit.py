from nbtlib import File
from nbtlib.tag import Compound, List, String
import json
import copy

nbt_path = r"C:\Users\34670\Downloads\MC-AI-BuildingGenerate-main\MC_test\yoasobi-runintonight.nbt"
# name = nbt_path.split("\\")[-1].rsplit(".", 1)[0]
name = "runintonight"

try:
    nbt = File.load(nbt_path, gzipped=True)
except:
    try:
        nbt = File.load(nbt_path, gzipped=False)
    except Exception as e:
        print(f"加载失败: {e}")
        exit()

def nbt_to_dict(tag):
    if isinstance(tag, Compound):
        return {key: nbt_to_dict(value) for key, value in tag.items()}
    elif isinstance(tag, List):
        return [nbt_to_dict(item) for item in tag]
    elif isinstance(tag, String):
        return str(tag)
    else:
        return tag

print("文件结构预览:", json.dumps(nbt_to_dict(nbt), indent=2)[:500] + "...")

def modify_commands(data, edition='JE', path=""):
    modified = False
    if isinstance(data, Compound):
        for key, value in list(data.items()):
            new_path = f"{path}.{key}" if path else key
            if key == 'Command':
                command = str(value)
                if "playsound" in command.lower():
                    if edition == 'JE':
                        if not command.lower().startswith("execute"):
                            new_command = f"execute as @a at @s run {command}"
                            data[key] = String(new_command)
                            print(f"[JE] 在 {new_path} 修改指令为: {new_command}")
                            modified = True
                    elif edition == 'BE':
                        if not command.lower().startswith("execute"):
                            command = f"execute as @a[tag={name}] at @s run {command}"
                            # command = f"execute as @a at @s run {command}"
                        # 替换音效标识
                        command = command.replace("minecraft:block.note_block.", "note.")
                        # 删除 playsound 后面的 `block`
                        parts = command.split()
                        if "playsound" in parts:
                            i = parts.index("playsound")
                            # 删除 source 为 block 的项（只能是 playsound xxx block ...）
                            if len(parts) > i + 2 and parts[i+2] == "block":
                                del parts[i+2]
                        command = " ".join(parts)
                        data[key] = String(command)
                        print(f"[BE] 在 {new_path} 修改指令为: {command}")
                        modified = True
            else:
                if modify_commands(value, edition, new_path):
                    modified = True
    elif isinstance(data, List):
        for i, item in enumerate(data):
            if modify_commands(item, edition, f"{path}[{i}]"):
                modified = True
    return modified

# JE 输出
nbt_je = copy.deepcopy(nbt)
modified_je = modify_commands(nbt_je, edition='JE')
if modified_je:
    output_je = nbt_path.replace(".nbt", "_modified_je.nbt")
    nbt_je.save(output_je, gzipped=True)
    print(f"[JE] 修改已保存到: {output_je}")
else:
    print("[JE] 未找到需要修改的 playsound 指令")

# BE 输出
nbt_be = copy.deepcopy(nbt)
modified_be = modify_commands(nbt_be, edition='BE')
if modified_be:
    output_be = nbt_path.replace(".nbt", "_modified_be.nbt")
    nbt_be.save(output_be, gzipped=True)
    print(f"[BE] 修改已保存到: {output_be}")
else:
    print("[BE] 未找到需要修改的 playsound 指令")
