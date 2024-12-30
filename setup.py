from setuptools import setup, find_packages

setup(
    name="pdf2anki",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfminer.six",
        "pytest",
        "fitz",
        "dataclasses; python_version<'3.7'", 
        # ...other dependencies...
    ],
    entry_points={
        "console_scripts": [
            # Example: "pdf2anki=pdf2anki.cli:main"
        ]
    },
)
