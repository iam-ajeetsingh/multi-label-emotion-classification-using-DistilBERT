# Install dependencies
install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

# Run tests
test:
	python -m pytest -vv test_*.py

# Train the model
train:
	python3 nn.py train

# Make predictions
predict:
	python3 nn.py predict

# Format code
format:
	black *.py

# Lint code
lint:
	pylint --disable=R,C *.py

# Run all tasks
all: install lint test format train predict
