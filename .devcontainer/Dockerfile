FROM registry.gitlab.com/islandoftex/images/texlive:latest

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ssh gnupg2 sudo wget && \
    apt-get clean && \
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
