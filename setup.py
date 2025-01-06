from setuptools import setup, find_packages

setup(
    name="mypdf2anki",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfminer.six",
        "pytest",
        "fitz",
        "dataclasses; python_version<'3.7'", 

    ],
    entry_points={
        "console_scripts": [
            "pdf2anki = pdf2anki.__main__:main"
        ]
    },
)