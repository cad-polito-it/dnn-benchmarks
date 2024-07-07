FROM tensorflow/tensorflow:2.14.0-gpu

COPY benchmark_models ./benchmark_models
COPY fault_lists ./fault_lists
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the start-script.sh into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Make sure the script is executable
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]