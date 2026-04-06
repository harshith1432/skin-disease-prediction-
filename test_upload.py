import os
import requests
from pathlib import Path

BASE = Path(__file__).parent
# find an image
imgs = list((BASE / 'model' / 'train').rglob('*.jpg')) + list((BASE / 'model' / 'train').rglob('*.jpeg')) + list((BASE / 'model' / 'train').rglob('*.png'))
if not imgs:
    print('No images found')
    raise SystemExit(1)
img = imgs[0]
print('Using image:', img)

s = requests.Session()
login = s.post('http://127.0.0.1:5000/login', data={'username':'admin','password':'password'})
print('Login status:', login.status_code)
resp = s.post('http://127.0.0.1:5000/predict', files={'file': open(img, 'rb')})
print('Predict status:', resp.status_code)
try:
    print(resp.json())
except Exception as e:
    print('Response content:', resp.text)
