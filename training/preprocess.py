import xml.etree.ElementTree as ET
import json
from pathlib import Path

# Pasta contendo N arquivos XML
xml_dir = Path(__file__).parent / ".." / "data" / "raw"
xml_dir = xml_dir.resolve()

# Arquivo JSONL final
jsonl_path = Path(__file__).parent / ".." / "data" / "processed" / "dataset_finetuning.jsonl"
jsonl_path = jsonl_path.resolve()

json_list = []

# Iterar sobre todos os arquivos XML da pasta
for xml_path in xml_dir.glob("*.xml"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Metadados do documento
    doc_id = root.attrib.get("id")
    source = root.attrib.get("source")
    url = root.attrib.get("url")
    focus = root.findtext("Focus")
    cuids = [cui.text for cui in root.findall(".//CUIs/CUI")]
    semantic_types = [st.text for st in root.findall(".//SemanticTypes/SemanticType")]
    semantic_group = root.findtext(".//SemanticGroup")

    # Iterar sobre cada QAPair
    for qapair in root.findall(".//QAPair"):
        json_list.append({
            "doc_id": doc_id,
            "source": source,
            "url": url,
            "focus": focus,
            "cuids": cuids,
            "semantic_types": semantic_types,
            "semantic_group": semantic_group,
            "pid": qapair.attrib.get("pid"),
            "question": qapair.findtext("Question"),
            "answer": qapair.findtext("Answer")
        })

# Salvar como JSONL
with open(jsonl_path, "w", encoding="utf-8") as f:
    for item in json_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Dataset JSONL gerado com sucesso em: {jsonl_path}")
print(f"📄 Total de registros: {len(json_list)}")
