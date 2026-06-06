

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

si daca conectezi prin USB esp-ul (asa recomand)

```bash
python esp32_bridge.py --port COM3 --name squat_reference
```