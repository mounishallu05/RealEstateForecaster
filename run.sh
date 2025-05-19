#!/bin/bash

# Simple script to run the house price prediction project

# Set up color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== US House Price Prediction Project ===${NC}"
echo

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

# Check if requirements are installed
echo -e "${YELLOW}Checking requirements...${NC}"
if ! command_exists pip3; then
    echo -e "${RED}Error: pip3 is not installed.${NC}"
    exit 1
fi

# Install requirements if needed
echo -e "${YELLOW}Installing requirements...${NC}"
pip3 install -r requirements.txt

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p data
fi

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p models
fi

# Check if data exists, if not generate it
if [ ! -f "data/housing_data.csv" ]; then
    echo -e "${YELLOW}Generating synthetic data...${NC}"
    python3 src/data_download.py
fi

# Display menu
echo -e "\n${GREEN}Choose an option:${NC}"
echo "1) Train a model"
echo "2) Evaluate an existing model"
echo "3) Start the web app"
echo "4) Run the full pipeline (generate data, train, evaluate, start app)"
echo "5) Exit"

read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Select model type:${NC}"
        echo "1) XGBoost (default)"
        echo "2) Random Forest"
        echo "3) LightGBM"
        echo "4) Elastic Net"
        echo "5) Ensemble (multiple models)"
        
        read -p "Enter model type [1-5]: " model_choice
        
        case $model_choice in
            1) model_type="xgboost";;
            2) model_type="random_forest";;
            3) model_type="lightgbm";;
            4) model_type="elastic_net";;
            5) model_type="ensemble";;
            *) model_type="xgboost";;
        esac
        
        echo -e "\n${YELLOW}Hyperparameter tuning?${NC}"
        read -p "Enable hyperparameter tuning? (y/n): " tune_choice
        
        if [[ $tune_choice == "y" || $tune_choice == "Y" ]]; then
            tune_flag="--tune-hyperparams"
        else
            tune_flag=""
        fi
        
        echo -e "\n${YELLOW}Training ${model_type} model...${NC}"
        python3 src/train.py --model-type ${model_type} ${tune_flag}
        ;;
        
    2)
        echo -e "\n${YELLOW}Evaluating model...${NC}"
        python3 src/evaluation.py
        ;;
        
    3)
        echo -e "\n${YELLOW}Starting web app...${NC}"
        python3 app.py
        ;;
        
    4)
        echo -e "\n${YELLOW}Running full pipeline...${NC}"
        
        # Generate data if needed
        if [ ! -f "data/housing_data.csv" ]; then
            echo -e "\n${YELLOW}Generating synthetic data...${NC}"
            python3 src/data_download.py
        else
            echo -e "\n${YELLOW}Using existing data...${NC}"
        fi
        
        # Train model
        echo -e "\n${YELLOW}Training ensemble model...${NC}"
        python3 src/train.py --model-type ensemble
        
        # Evaluate model
        echo -e "\n${YELLOW}Evaluating model...${NC}"
        python3 src/evaluation.py
        
        # Start web app
        echo -e "\n${YELLOW}Starting web app...${NC}"
        python3 app.py
        ;;
        
    5)
        echo -e "\n${GREEN}Exiting.${NC}"
        exit 0
        ;;
        
    *)
        echo -e "\n${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac 