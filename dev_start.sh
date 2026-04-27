#!/bin/bash
# dev_start.sh - Start a new LeWorldModel Docker container (creates container only, does not enter)

set -e

# Configuration
IMAGE_NAME="le-wm:latest"
CONTAINER_NAME="le-wm-container"
PROJECT_DIR="$(pwd)"
STORAGE_DIR="/storage"

# Host data directory for models and datasets
# You can change this to any directory on your host machine
HOST_DATA_DIR="${PROJECT_DIR}/.stable-wm"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed, please install Docker first"
        exit 1
    fi
}

# Check NVIDIA Docker runtime
check_nvidia_docker() {
    if ! docker info | grep -q "Runtimes:.*nvidia"; then
        print_warning "NVIDIA Docker runtime not detected"
        print_warning "If using GPU, please install nvidia-docker2"
        echo ""
        echo "Installation:"
        echo "  Ubuntu/Debian:"
        echo "    sudo apt-get install nvidia-docker2"
        echo "    sudo systemctl restart docker"
        echo ""
        USE_GPU="false"
    else
        USE_GPU="true"
    fi
}

# Create necessary directories
create_directories() {
    # Create host data directory if not exists
    mkdir -p "$HOST_DATA_DIR"
    print_success "Host data directory: $HOST_DATA_DIR"
    
    # Create subdirectories for data and models
    mkdir -p "$HOST_DATA_DIR/datasets"
    mkdir -p "$HOST_DATA_DIR/checkpoints"
    mkdir -p "$HOST_DATA_DIR/logs"
    print_success "Created subdirectories: datasets, checkpoints, logs"

    # Check storage directory
    if [ -d "$STORAGE_DIR" ]; then
        print_success "Storage directory exists: $STORAGE_DIR"
    else
        print_warning "Storage directory does not exist: $STORAGE_DIR"
    fi
}

# Check if container already exists
check_existing_container() {
    # Check if container exists (running or stopped)
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        # Check if container is running
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            print_info "Container '$CONTAINER_NAME' is already running"
            print_info "Use ./dev_into.sh to enter"
            exit 0
        else
            # Container exists but is stopped
            print_warning "Container '$CONTAINER_NAME' exists but is stopped"
            print_info "Starting container..."
            docker start "$CONTAINER_NAME"
            print_success "Container started"
            print_info "Use ./dev_into.sh to enter"
            exit 0
        fi
    fi
}

# Check if image exists
check_image() {
    if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
        print_error "Image '$IMAGE_NAME' does not exist"
        print_info "Please build the image first:"
        echo "  docker build -t $IMAGE_NAME ."
        exit 1
    fi
}

# Start container in detached mode
start_container() {
    print_info "Starting new LeWorldModel Docker container in detached mode..."

    # Base Docker run parameters (detached mode)
    DOCKER_RUN_CMD="docker run -itd"

    # Container name
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD --name $CONTAINER_NAME"

    # GPU support
    if [ "$USE_GPU" = "true" ]; then
        DOCKER_RUN_CMD="$DOCKER_RUN_CMD --gpus all"
        print_info "GPU support enabled"
    fi

    # Shared memory
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD --shm-size=16gb"

    # Volume mounts - mount host data directory to container
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
        -v $PROJECT_DIR:/workspace/le-wm \
        -v ${HOST_DATA_DIR}:/workspace/.stable-wm \
        -v $STORAGE_DIR:$STORAGE_DIR"

    # Network settings (for Jupyter and TensorBoard)
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -p 8888:8888 -p 6006:6006"

    # Environment variables
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD \
        -e PYTHONUNBUFFERED=1 \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e STABLEWM_HOME=/workspace/.stable-wm"

    # Add image name
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD $IMAGE_NAME /bin/bash"

    # Print startup info
    echo ""
    print_info "Start command:"
    echo "$DOCKER_RUN_CMD"
    echo ""
    print_info "Container configuration:"
    echo "  - Container name: $CONTAINER_NAME"
    echo "  - Image name: $IMAGE_NAME"
    echo "  - GPU support: $USE_GPU"
    echo "  - Shared memory: 16GB"
    echo "  - Port mapping: 8888 (Jupyter), 6006 (TensorBoard)"
    echo ""
    print_info "Directory mounts:"
    echo "  - Project: $PROJECT_DIR -> /workspace/le-wm"
    echo "  - Data/Models (Host): ${HOST_DATA_DIR} -> /workspace/.stable-wm"
    echo "  - Storage: $STORAGE_DIR -> $STORAGE_DIR"
    echo ""
    print_info "Host data directory structure:"
    echo "  ${HOST_DATA_DIR}/"
    echo "  ├── datasets/     # Store HDF5 datasets here"
    echo "  ├── checkpoints/  # Model checkpoints will be saved here"
    echo "  └── logs/         # Training logs"
    echo ""

    # Execute start command
    eval "$DOCKER_RUN_CMD"

    echo ""
    print_success "Container '$CONTAINER_NAME' started in background"
    echo ""
    print_info "To enter the container, run:"
    echo "  ./dev_into.sh"
    echo ""
    print_info "To view container status:"
    echo "  docker ps"
    echo ""
    print_info "To view container logs:"
    echo "  docker logs $CONTAINER_NAME"
    echo ""
}

# Main function
main() {
    echo "========================================="
    echo "   LeWorldModel Docker Container Start Script"
    echo "   (Creates container only, does not enter)"
    echo "========================================="
    echo ""

    check_docker
    check_nvidia_docker
    create_directories
    check_image
    check_existing_container

    # If not exited, start new container
    start_container
}

# Run main function
main
