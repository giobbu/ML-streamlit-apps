FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml environment.yml
RUN conda env create -f environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

EXPOSE 8501

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "graph-app", "streamlit", "run", "main.py"]