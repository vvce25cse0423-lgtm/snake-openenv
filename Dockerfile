FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install gymnasium numpy
EXPOSE 7860
CMD ["python", "server.py"]