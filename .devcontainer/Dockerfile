FROM registry.gitlab.com/islandoftex/images/texlive@sha256:e51e9458a2c18970d6661c3a858c4b069de45a35ff40ecc5f8a0dfb6e53aaff4

ARG ENV_NAME=base
ARG USER_NAME=texlive

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

# Add user to sudo group
RUN usermod -aG sudo $USER_NAME

# Switch to user
USER $USER_NAME

# Set user environment variables
ENV CONDA_ROOT="/home/$USER_NAME/miniconda3"

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "$CONDA_ROOT" && \
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    rm /tmp/miniconda.sh

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN sudo chown -R $USER_NAME:$USER_NAME /tmp/conda-tmp && \
    if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
        umask 0002 && \
        $CONDA_ROOT/bin/conda env create -y -n $ENV_NAME -f /tmp/conda-tmp/environment.yml && \
        $CONDA_ROOT/bin/conda clean -afy; \
    else \
        echo "No environment.yml found, skipping conda environment creation"; \
    fi \
    && rm -rf /tmp/conda-tmp

# Activate conda environment by default
RUN echo "source $CONDA_ROOT/etc/profile.d/conda.sh" >> /home/$USER_NAME/.bashrc && \
    echo "conda activate $ENV_NAME" >> /home/$USER_NAME/.bashrc && \
    echo "auto_activate: true" >> /home/$USER_NAME/.condarc && \
    echo "default_activation_env: $ENV_NAME" >> /home/$USER_NAME/.condarc
