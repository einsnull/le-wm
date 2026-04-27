#!/bin/bash
# dev_into.sh - Enter an existing LeWorldModel container

set -e

# Configuration
CONTAINER_NAME="le-wm-container"
STORAGE_DIR="/storage"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if container exists
check_container_exists() {
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "Container '$CONTAINER_NAME' does not exist"
        print_info "Please create the container first: ./dev_start.sh"
        exit 1
    fi
}

# Check if container is running
check_container_running() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container '$CONTAINER_NAME' is not running"
        read -p "Start the container? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker start "$CONTAINER_NAME"
            print_success "Container started"
        else
            exit 0
        fi
    fi
}

# Display container status
show_container_status() {
    docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Enter container
enter_container() {
    print_info "Entering container '$CONTAINER_NAME'..."
    print_info "Available commands:"
    echo "  - Train: python train.py data=pusht"
    echo "  - Eval:  python eval.py --config-name=pusht.yaml policy=pusht/lewm"
    echo "  - Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    echo "  - Start TensorBoard: tensorboard --logdir=/workspace/.stable-wm --bind_all"
    echo "  - Access storage: cd $STORAGE_DIR"
    echo "  - Exit container: exit"
    echo ""
    print_info "Data directories (persisted on host):"
    echo "  - Datasets:   /workspace/.stable-wm/datasets/"
    echo "  - Checkpoints:/workspace/.stable-wm/checkpoints/"
    echo "  - Logs:       /workspace/.stable-wm/logs/"
    echo ""

    docker exec -it \
        -e STABLEWM_HOME=/workspace/.stable-wm \
        "$CONTAINER_NAME" \
        /bin/bash

    echo ""
    print_info "Exited container '$CONTAINER_NAME'"
    show_container_status
}

# Main function
main() {
    echo "========================================="
    echo "   Enter LeWorldModel Docker Container"
    echo "========================================="
    echo ""

    check_container_exists
    check_container_running
    enter_container
}

main
