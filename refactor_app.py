import os

app_path = "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if line.startswith("@st.cache_resource") and i+1 < len(lines) and "def load_beit_model" in lines[i+1]:
        start_idx = i
        break

end_idx = -1
if start_idx != -1:
    for i in range(start_idx, len(lines)):
        if "def main():" in lines[i]:
            end_idx = i
            break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + ["\n"] + lines[end_idx:]
    with open(app_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Removed lines {start_idx} to {end_idx-1}")
else:
    print(f"Could not find indices: start={start_idx}, end={end_idx}")
