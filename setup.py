from setuptools import setup, find_packages

setup(
    name="nupe_ng_tts",
    version="0.1.0",
    description="Neural text-to-speech for the Nupe language (seq2seq LSTM + attention)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "soundfile>=0.11.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
)
