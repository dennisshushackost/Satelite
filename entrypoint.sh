#!/bin/bash
# Fix permissions
chown -R tfuser:tfuser /home/tfuser/project
# Execute the Docker CMD
exec "$@"
