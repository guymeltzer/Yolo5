FROM ultralytics/yolov5:v6.2-cpu
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt
COPY check_python.sh /app/check_python.sh
RUN chmod +x /app/check_python.sh
RUN /app/check_python.sh
RUN mkdir -p data && curl -L https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml -o data/coco128.yaml
# Add CA certificate
COPY ca.crt /app/ca.crt
COPY . .
CMD ["python3", "app.py"]