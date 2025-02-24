
# Eternal-Coders_AI14 (AI crop disease prediction and management)

Eternal-Coders_AI14 (AI crop disease prediction and management) is a plant disease detection application that leverages deep learning techniques. The application uses TensorFlow's MobileNetV2 model to identify plant diseases from leaf images. It also integrates with MongoDB for data storage.

## Features
- Upload leaf images to detect plant diseases.
- Uses MobileNetV2 for efficient image classification.
- Web interface built with Flask.
- Stores results in MongoDB.

## Prerequisites
- Python 3.x
- MongoDB

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Yashwanth1304/Eternal-Coders_AI14.git
    cd Eternal-Coders_AI14
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow flask Pillow opencv-python-headless numpy pymongo
    ```

3. Ensure MongoDB is running on your system.

## Usage

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Open your browser and go to `http://localhost:5000` to use the application.

## Project Structure
```
Eternal-Coders_AI14/
│   app.py                    # Main application file
│   model_development.ipynb   # Jupyter notebook for model development
│   requirements.txt          # List of dependencies
│
├── static/                   # Static image assets
│   └── ...                   # Leaf images for disease detection
├── plant disease/            # Diseased plant images
│   └── ... 
└── templates/                # HTML templates for the web interface
    └── ...                   # HTML files for Flask routing
```

## Dependencies
- Flask
- TensorFlow
- Pillow
- OpenCV
- NumPy
- PyMongo

## Acknowledgements
This project is developed by our team Eternal Coders as part of Hackathon. Special thanks to the team for their dedication.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
