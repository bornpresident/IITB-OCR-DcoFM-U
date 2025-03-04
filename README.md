# OCR for DocFM

This repository contains an Optical Character Recognition (OCR) pipeline specifically curated for scientific documents, designed to pre-train a foundational model for document understanding (DocFM). The original work was developed by Dhruv (MS Research, IIT Bombay), and this repository includes updates and improvements.

## Features
- Optimized OCR pipeline for scientific documents
- Supports multiple languages
- Streamlit-based UI for easy interaction
- Output visualization for better understanding
- Modular and customizable setup

## Installation
It is recommended to use a virtual environment before installing dependencies.

```sh
pip install -r requirements.in
```

## Usage

### Step 1: Running the Pipeline
Execute the `main.py` script to set input parameters, define the output name, and specify the language.

```sh
python3 main.py
```

### Step 2: Using the Streamlit UI
A Streamlit-based UI is available for easier execution and downloading the processed output.

```sh
streamlit run app.py
```

### Step 3: Visualizing the Results
You can visualize the processed output using the Streamlit viewer. Ensure that HTML generation is enabled in the `CorrectorOutput` folder (currently disabled).

```sh
streamlit run viewer.py
```

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests to improve the pipeline.

## License
This project is licensed under [insert appropriate license].

---
For further questions, feel free to reach out or create an issue in this repository.

