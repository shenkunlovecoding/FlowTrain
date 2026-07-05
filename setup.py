from pathlib import Path

from setuptools import find_packages, setup


readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""


setup(
    name="flowtrain",
    version="0.1.0",
    description="FlowTrain: RWKV-7 training with CPU master weights and TileLang backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["flowtrain", "flowtrain.*"]),
    package_data={"flowtrain": ["csrc/*.cpp"]},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "tilelang",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "deepspeed": ["deepspeed"],
        "sft": ["transformers", "pyrwkv_tokenizer"],
    },
    entry_points={
        "console_scripts": [
            "flowtrain-train-rwkv7=flowtrain.cli.train_rwkv7:main",
            "flowtrain-train-sft=flowtrain.cli.train_sft:main",
            "flowtrain-estimate-rwkv7-bs=flowtrain.cli.estimate_rwkv7_bs:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
