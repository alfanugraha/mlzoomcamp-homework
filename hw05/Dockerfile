FROM agrigorev/zoomcamp-model:2025
WORKDIR /code

COPY pyproject.toml pyproject.toml
RUN pip install .

COPY load_fastapi.py load_fastapi.py
COPY pipeline_v1.bin pipeline_v1.bin
COPY test.py test.py

EXPOSE 9696
ENTRYPOINT ["uvicorn", "load_fastapi:app", "--host", "0.0.0.0", "--port", "9696"]