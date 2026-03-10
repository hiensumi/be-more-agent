import urllib.request
url = 'https://html-classic.itch.zone/html/13484643/D:/BmoV1.1/index.html'
with urllib.request.urlopen(url) as response:
    html = response.read().decode('utf-8')
with open('dl_index.html', 'w', encoding='utf-8') as f:
    f.write(html)
