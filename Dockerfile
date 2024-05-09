# Execute by: docker build -t satellite-project .
# Bind: docker run -d -p 2222:22 --gpus all --name my-tf-container -v /path/to/your/local/project:/project satellite-project
# Bind: docker run -d -p 2222:22 --gpus all --name my-tf-container -v C:/Users/dshus/Desktop/project:/home/tf/project satellite-project
# SSH into the docker container: ssh -p 2222 tfuser@localhost

# Use the official TensorFlow GPU image as the base
FROM tensorflow/tensorflow:latest-gpu

# Install SSH server and any other necessary packages
RUN apt-get update && apt-get install -y \
    openssh-server \
    vim \
    git \
    sudo  # Ensure sudo is installed

# Setup SSH server
RUN mkdir /var/run/sshd

# Create a non-root user 'tfuser' and set a password, also add to sudoers
RUN useradd -m tfuser && echo "tfuser:password" | chpasswd
RUN echo "tfuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Setup user's home directory and SSH accessibility
RUN mkdir -p /home/tfuser/.ssh && chown -R tfuser:tfuser /home/tfuser/.ssh && chmod 700 /home/tfuser/.ssh

# Change SSH settings to allow the new user to login
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Set environment variable
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Expose port 22 for SSH access
EXPOSE 22 

# Install additional Python libraries
RUN pip install numpy scipy matplotlib geopandas rasterio scikit-learn eodal scikit-image jupyter

# Copy files from your project folder into the container
COPY . /home/tfuser/project

# Set the working directory and change ownership to 'tfuser'
WORKDIR /home/tfuser/project
RUN chown -R tfuser:tfuser /home/tfuser/project

# Switch to user 'tfuser'
USER tfuser

# Generate SSH keys for 'tfuser' (optional, if needed for SSH into other servers)
RUN ssh-keygen -t rsa -b 4096 -C "dennis.shushack@ost.ch" -f /home/tfuser/.ssh/id_rsa -N "" && \
    chmod 400 /home/tfuser/.ssh/id_rsa

# Command to start services, needs to switch back to root to start SSHD
USER root
CMD ["/usr/sbin/sshd", "-D"]
