#!/bin/bash

echo "Installing required SDL2 C++ Development Headers for Raspberry Pi..."
sudo apt-get update
sudo apt-get install -y libsdl2-dev libsdl2-mixer-dev g++ make

echo "Compiling Guardians of Sunshine from source..."
make clean
make

echo "Done! You can now run the game using: ./GuardiansOfSunshine"
