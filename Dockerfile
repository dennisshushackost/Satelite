# Execute by: docker build -t satellite-project .
# Bind: docker run -d -p 2222:22 --gpus all --name my-tf-container -v /path/to/your/local/project:/project satellite-project
# Bind: docker run -d -p 2222:22 --gpus all --name my-tf-container -v C:/Users/dshus/Desktop/project:/project satellite-project
# SSH into the docker container: ssh -p 2222 root@localhost

# Use the official TensorFlow GPU image as the base
FROM tensorflow/tensorflow:latest-gpu

# Install SSH server and any other packages
RUN apt-get update && apt-get install -y \
    openssh-server \
    vim \
    git

# Setup SSH server
RUN mkdir /var/run/sshd
# Replace 'password' with your chosen password or use an environment variable
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Expose port 22 for SSH access
EXPOSE 22

# Install additional Python libraries
RUN pip install numpy scipy matplotlib geopandas rasterio scikit-learn eodal scikit-image

# Copy files from your project folder into the container
COPY . /project

# Set the working directory
WORKDIR /project

# Generate SSH keys and set permissions
RUN ssh-keygen -t rsa -b 4096 -C "dennis.shushack@ost.ch" -f /root/.ssh/id_rsa -N "" && \
    chmod 400 /root/.ssh/id_rsa

# Command to start services
CMD ["/usr/sbin/sshd", "-D"]
