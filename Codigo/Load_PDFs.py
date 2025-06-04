import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import urllib3

# Desactiva advertencias SSL si usamos verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

base_output_path = r'C:/Users/David/Desktop/Chatbot/Datos'  # ← ruta corregida a absoluta

def descargar_con_tolerancia(url):
    try:
        return requests.get(url, timeout=10)
    except requests.exceptions.SSLError:
        print(f"⚠️ SSL error en {url}, reintentando sin verificación...")
        try:
            return requests.get(url, timeout=10, verify=False)
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            return None
    except Exception as e:
        print(f"❌ Error general: {e}")
        return None

def download_pdf(pdf_url, output_folder):
    try:
        filename = os.path.basename(pdf_url)
        if not filename.lower().endswith('.pdf'):
            filename += ".pdf"
        filename = os.path.join(output_folder, filename)

        if not os.path.exists(filename):
            print(f'📥 Descargando PDF: {pdf_url}')
            r = descargar_con_tolerancia(pdf_url)
            if r and r.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f'✅ PDF guardado como: {filename}')
            else:
                print(f'❌ No se pudo descargar PDF: {pdf_url}')
        else:
            print(f'⚠️ PDF ya existe: {filename}')
    except Exception as e:
        print(f'❌ Error al descargar {pdf_url}: {e}')

def process_page(url):
    last_path = urlparse(url).path.split('/')[-1]
    folder_name = last_path if last_path else "Documento"
    output_folder = os.path.join(base_output_path, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    response = descargar_con_tolerancia(url)
    if not response or response.status_code != 200:
        print(f'❌ Error accediendo a {url}')
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Guardar solo texto visible
    for a in soup.find_all('a'):
        a.decompose()

    page_text = soup.get_text(separator='\n', strip=True)
    text_file_path = os.path.join(output_folder, f'{last_path}_texto.txt')
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(page_text)
    print(f'📝 Texto guardado en: {text_file_path}')

    # Buscar enlaces a PDF
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        full_url = urljoin(url, href)
        if ('pdf' in full_url.lower()) or re.search(r"SFS\\d+", full_url):
            download_pdf(full_url, output_folder)

    print(f'✅ Finalizado: {url}\\n')

# Leer URLs desde archivo
with open('enlaces.txt', 'r', encoding='utf-8') as f:
    urls = [line.strip() for line in f if line.strip()]

# Procesar todas
for url in urls:
    process_page(url)
