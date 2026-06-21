# Hanzi-CNN-Recognition

A deep learning project for handwritten Chinese character recognition using Convolutional Neural Networks (CNN) with PyTorch. This project covers the full pipeline from data preprocessing to real-time inference.

## Project Overview
This project provides a complete workflow for training and deploying a Hanzi recognition model:
- **Data Pipeline**: Preprocessing, augmentation, and HDF5 conversion for efficient training.
- **Model Training**: CNN architecture definition and training process using Jupyter Notebook.
- **Web Demo**: Interactive recognition interface built with Gradio.

## Dataset
The dataset used for training this model was sourced from [Handwritten Chinese Character (Hanzi) Datasets].
- You can access the raw data here: [Kaggle - Hanzi Dataset](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets/data)
- The data is processed using `setup_data.py` to extract and convert it into HDF5 format for PyTorch compatibility.

## Project Structure
- `setup_data.py`: Handles data preprocessing and conversion to HDF5 format.
- `data_pipeline.py`: Contains data augmentation strategies and custom DataLoaders.
- `model.ipynb`: Jupyter notebook for CNN architecture and model training.
- `app.py`: Gradio application for real-time model inference.
- `mapping.json`, `mapping_hsk.json`, `mapping_pinyin.json`: Lookup files for character information, HSK levels, and Pinyin.
- `requirements.txt`: Project dependencies.

## Tech Stack
- **Python**: Core programming language.
- **PyTorch**: Deep learning framework.
- **Gradio**: Web-based demo interface.
- **HDF5 (h5py)**: Data storage for large datasets.

## How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NgoQuangTao2005/Hanzi-CNN-Recognition.git](https://github.com/NgoQuangTao2005/Hanzi-CNN-Recognition.git)

2. **Install dependencies:**
In Terminal
   ```bash
   pip install -r requirements.txt

4. **Launch the Demo:**
   ```bash
   python app.py
The Gradio interface will launch in your browser automatically.

## Notes
- **Data Preparation**: Run `setup_data.py` first to generate the necessary data format if you are training from scratch.
- **Training**: Use `model.ipynb` to train the CNN model. Ensure your dataset is structured correctly before running the training notebook.
- **Path Configuration**: If you encounter `FileNotFoundError`, please double-check the paths. Make sure the paths to your dataset and weight files match your local machine's directory structure.

Created by Ngo Quang Tao
