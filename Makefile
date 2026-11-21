.PHONY: install build-model quantize test benchmark run docker-build docker-run clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

build-model:
	$(PYTHON) -m src.models.builder --output models/defect_detector_fp32.onnx

quantize:
	$(PYTHON) -m src.models.quantizer --input models/defect_detector_fp32.onnx --output models/defect_detector_int8.onnx

test:
	SIMULATION_MODE=true pytest -v

benchmark:
	$(PYTHON) benchmarks/benchmark_fps.py

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t edgevision-int8:latest .

docker-run:
	docker run -p 8000:8000 --rm --name edgevision-service edgevision-int8:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage models/*.onnx
