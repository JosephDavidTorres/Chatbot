#pip install requests beautifulsoup4
'Me ayuda a obtener todos los documentos PDF de los enlaces de legislación'
import os
import requests
from bs4 import BeautifulSoup       #Lo utilizo para visitar la pagina y recapitular toda la informacion que necesito
from urllib.parse import urljoin, urlparse
import re  # Importamos la librería para expresiones regulares

# Ruta raíz donde voy guardar todo
base_output_path = r'/Datos'

# Función para descargar los PDFs
def download_pdf(pdf_url, output_folder):
    try:
        # Si la URL no tiene una extensión, se asegura de que el archivo se guarde como .pdf
        filename = os.path.basename(pdf_url)
        if not filename.lower().endswith('.pdf'):
            filename = filename + '.pdf'  # Fuerza la extensión .pdf

        filename = os.path.join(output_folder, filename)

        if not os.path.exists(filename):
            print(f'📥 Descargando PDF: {pdf_url}')
            with requests.get(pdf_url, stream=True, timeout=10) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f'✅ PDF guardado como: {filename}')
        else:
            print(f'⚠️ PDF ya existe: {filename}')
    except Exception as e:
        print(f'❌ Error al descargar {pdf_url}: {e}')

# Función para procesar la página
def process_page(url):
    # Última parte de la ruta para nombrar la carpeta
    last_path = urlparse(url).path.split('/')[-1]
    folder_name = last_path if last_path else "Documento"

    # Carpeta final completa: base + nombre de la página
    output_folder = os.path.join(base_output_path, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Obtener el HTML de la página
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Eliminar enlaces para dejar solo el texto
        for a in soup.find_all('a'):
            a.decompose()

        # Extraer el texto
        page_text = soup.get_text(separator='\n', strip=True)

        # Guardar texto en archivo
        text_file_path = os.path.join(output_folder, f'{last_path}_texto.txt')
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(page_text)

        print(f'📝 Texto guardado en: {text_file_path}')

        # Volver a analizar para buscar enlaces a PDF o con SFS seguido de números
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            full_url = urljoin(url, href)

            # Filtrar enlaces que contengan "pdf" o que contengan "SFS" seguido de números
            if ('pdf' in full_url.lower()) or re.search(r"SFS\d+", full_url):
                download_pdf(full_url, output_folder)

        print(f'✅ Finalizado: {url}\n')

    except Exception as e:
        print(f'❌ Error procesando {url}: {e}')

# Leer todas las URLs desde el archivo
with open('enlaces.txt', 'r', encoding='utf-8') as f:
    urls = [line.strip() for line in f if line.strip()]

# Procesar todas las URLs
for url in urls:
    process_page(url)
