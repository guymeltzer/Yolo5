# Use ultralytics/yolov5:v6.2-cpu as the base image
FROM ultralytics/yolov5:v6.2-cpu

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download YOLO model
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

# Run script to check Python installation and PATH
COPY check_python.sh /app/check_python.sh
RUN chmod +x /app/check_python.sh
RUN /app/check_python.sh

# Ensure the 'data' directory exists before downloading coco128.yaml
RUN mkdir -p data && \
    curl -L https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml -o data/coco128.yaml

# Copy application files
COPY . .

# Run command when container starts
CMD ["python3", "app.py"]
