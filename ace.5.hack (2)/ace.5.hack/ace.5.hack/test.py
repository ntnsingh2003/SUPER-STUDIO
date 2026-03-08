import requests

try:
    res = requests.post(
        'http://127.0.0.1:8000/api/v1/execute-workflow',
        json={'service': 'creative', 'prompt': 'What is the color of the sky?'}
    )
    print("STATUS:", res.status_code)
    print("RESPONSE:", res.text)
except Exception as e:
    print("ERROR:", e)
