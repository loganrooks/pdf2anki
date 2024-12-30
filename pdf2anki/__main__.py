import argparse
from pdf2anki import convert_pdf_to_anki

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to Anki deck.')
    parser.add_argument('pdf_path', help='Path to the PDF file to convert')
    parser.add_argument('-o', '--output', help='Output Anki deck file', default='output.apkg')
    args = parser.parse_args()

    convert_pdf_to_anki(args.pdf_path, args.output)
    print(f"Anki deck created at {args.output}")

if __name__ == '__main__':
    main()