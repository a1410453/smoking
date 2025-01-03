﻿# Smoking classification

This project is a FastAPI application that integrates YOLO for classification and OpenCV for image processing. Unfortunately, due to technical challenges, the Docker container setup is incomplete. However, the application can be run locally using Uvicorn.
## Input:
![uploading](image.png)

## Output:
{
    "smoking_timestamps": [
        4,
        5,
        6,
        7,
        8,
        9,
        13
    ]
}

There are example videos for test
![result](image-1.png)


---

## Features

- **FastAPI**: web framework for building API with Python.
- **YOLO**: model for classification top 1 acc. = 0.8849
- **OpenCV**: Utilized for image processing tasks.
---

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.12.7
- pip (Python package manager)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/a1410453/smoking
   cd your-repository
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add path for model:

in main.py, line 91:         
    model_path = "C:/Users/a1410/Downloads/smoking/runs/classify/train/weights/best.pt"

---

## Running the Application

To start the application, use the following command:

```bash
uvicorn main:app --reload
```

- The app will be available at `http://127.0.0.1:8000`

---

## Notes on Docker

The project includes a `Dockerfile` for containerization. Unfortunately, due to unresolved issues, the Docker container could not be successfully created. For now, the application can only be run locally.

---

## Future Improvements

- Resolve the Docker containerization issue.
- Enhance the deployment process for better scalability.
- Find better dataset, with more images.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
