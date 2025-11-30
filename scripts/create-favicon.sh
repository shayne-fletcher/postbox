#!/bin/bash
# Create favicon from ottie.jpg
# Requires ImageMagick: brew install imagemagick (Mac) or apt install imagemagick (Linux)

set -e

cd "$(dirname "$0")"

echo "Creating favicon from ottie.jpg..."

# Create different sizes
convert ottie.jpg -resize 32x32 -quality 100 favicon-32x32.png
convert ottie.jpg -resize 16x16 -quality 100 favicon-16x16.png

# Combine into .ico file
convert favicon-32x32.png favicon-16x16.png favicon.ico

# Clean up intermediate files
rm favicon-32x32.png favicon-16x16.png

echo "✓ Created favicon.ico"
echo "Now Claude can integrate it into the documentation!"
