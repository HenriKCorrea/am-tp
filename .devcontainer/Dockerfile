FROM registry.gitlab.com/islandoftex/images/texlive:latest

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg2 \
        git \
        git-lfs \
        ssh \
        sudo \
        wget \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create /etc/sudoers.d/001-sudo-group to enable passwordless sudoers group
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/001-sudo-group && \
    chmod 440 /etc/sudoers.d/001-sudo-group

# Add user texlive (created on textlive docker layer) to sudo group
RUN usermod -aG sudo texlive

# Switch to user texlive
USER texlive

# Install Miniconda
RUN mkdir -p ~/miniconda3 && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

# Copy environment.yml to a temp location to create environment.
COPY environment.yml /tmp/conda-tmp/

# Extract environment name from environment.yml
RUN ENV_NAME=$(awk '/^name:/ {print $2}' /tmp/conda-tmp/environment.yml) && \
    echo "Environment name is $ENV_NAME"

# Create conda environment
RUN ~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    umask 0002 && \
    ~/miniconda3/bin/conda env create -y -f /tmp/conda-tmp/environment.yml && \
    ~/miniconda3/bin/conda clean -afy

# Update PATH environment variable
ENV PATH="/home/texlive/miniconda3/envs/$ENV_NAME/bin:/home/texlive/miniconda3/bin:$PATH"

# Activate conda environment by default
RUN echo "source ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $ENV_NAME" >> ~/.bashrc
