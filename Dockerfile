FROM python:3.9

WORKDIR /code

# Copy requirements.txt file
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your application code
COPY main.py /code/main.py
COPY data/ /code/data/

# Run your application
CMD ["python", "main.py"]
