"""
Generate a self-contained HTML file with embedded images for PDF conversion.
Open the HTML in a browser and press Ctrl+P to Save as PDF.
"""
import base64
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'outputs')

def encode_image(filename):
    with open(os.path.join(OUTPUT_DIR, filename), 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

scatter_b64 = encode_image('scatter_bmi_vs_charges.png')
histogram_b64 = encode_image('histogram_charges.png')
heatmap_b64 = encode_image('correlation_heatmap.png')

# Read the HTML template
template_path = os.path.join(SCRIPT_DIR, 'template.html')
with open(template_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Replace placeholders with actual base64 images
html = html.replace('SCATTER_PLACEHOLDER', scatter_b64)
html = html.replace('HISTOGRAM_PLACEHOLDER', histogram_b64)
html = html.replace('HEATMAP_PLACEHOLDER', heatmap_b64)

output_path = os.path.join(OUTPUT_DIR, 'assignment_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print('HTML report saved:', output_path)
print()
print('To convert to PDF:')
print('   1. Open assignment_report.html in Google Chrome')
print('   2. Press Ctrl + P')
print('   3. Change Destination to "Save as PDF"')
print('   4. Set margins to "Default"')
print('   5. Enable "Background graphics"')
print('   6. Click Save')
