FROM continuumio/miniconda3


WORKDIR /app

COPY environment.yml environment.yml

RUN conda env create -f environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate tf-stream" > ~/.bashrc
ENV PATH /opt/conda/envs/tf-stream/bin:$PATH



EXPOSE 8501

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf-stream", "streamlit", "run", "run_.py"]

