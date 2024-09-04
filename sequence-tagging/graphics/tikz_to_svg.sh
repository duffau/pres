#!/bin/bash

# Check if the correct number of arguments is given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <tikz_file.tex> <output_file.svg>"
    exit 1
fi

# Input TikZ file and output SVG file
TIKZ_FILE=$1
OUTPUT_SVG=$2

# Get the current directory to return to later
CURRENT_DIR=$(pwd)

# Create a temporary directory to work in
TEMP_DIR=$(mktemp -d)

# Copy the TikZ file to the temporary directory
cp "$TIKZ_FILE" "$TEMP_DIR/tikzfile.tex"

# Change to the temporary directory
cd "$TEMP_DIR" || exit

# Compile the TikZ file to a PDF using pdflatex
pdflatex -interaction=nonstopmode tikzfile.tex

# Convert the resulting PDF to SVG
pdf2svg tikzfile.pdf tikzfile.svg

# Move the SVG file to the original directory
mv tikzfile.svg "$CURRENT_DIR/$OUTPUT_SVG"

# Return to the original directory
cd "$CURRENT_DIR" || exit

# Cleanup: remove the temporary directory
rm -rf "$TEMP_DIR"

echo "SVG saved as $OUTPUT_SVG"
